#include "server-trace.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

enum trace_kind {
    TRACE_KIND_NONE = 0,
    TRACE_KIND_EXPERTS,
    TRACE_KIND_WEIGHTS,
    TRACE_KIND_SHARED,
    TRACE_KIND_RESID,
};

// tensors are named "<prefix>-<il>" by the graph builder
static bool trace_match_layer(const char * name, const char * prefix, int32_t & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') {
        return false;
    }
    char * end = nullptr;
    const long v = strtol(name + n + 1, &end, 10);
    if (end == name + n + 1 || *end != '\0' || v < 0) {
        return false;
    }
    il = (int32_t) v;
    return true;
}

static int trace_classify(const char * name, int32_t & il) {
    if (trace_match_layer(name, "ffn_moe_topk", il)) {
        return TRACE_KIND_EXPERTS;
    }
    if (trace_match_layer(name, "ffn_moe_weights_norm", il)) {
        return TRACE_KIND_WEIGHTS;
    }
    if (trace_match_layer(name, "shared_expert_gate_sigmoid", il)) {
        return TRACE_KIND_SHARED;
    }
    if (trace_match_layer(name, "l_last", il)) {
        il += 1; // resid index
        return TRACE_KIND_RESID;
    }
    if (strcmp(name, "hc_init") == 0) {
        il = 0;
        return TRACE_KIND_RESID;
    }
    return TRACE_KIND_NONE;
}

void server_moe_trace::init(int32_t n_layer_) {
    n_layer = n_layer_;
    layers.assign(n_layer, layer_capture());
    resid.assign(n_layer + 1, resid_capture());
}

void server_moe_trace::install(llama_context * ctx) {
    if (!installed) {
        llama_set_eval_callback(ctx, eval_cb, this);
        installed = true;
    }
}

void server_moe_trace::remove(llama_context * ctx) {
    if (installed) {
        llama_set_eval_callback(ctx, nullptr, nullptr);
        installed = false;
    }
    begin_batch();
}

void server_moe_trace::begin_batch() {
    for (auto & l : layers) {
        l = layer_capture();
    }
    for (auto & r : resid) {
        r = resid_capture();
    }
    ubatch_base  = 0;
    ubatch_n_tok = 0;
}

bool server_moe_trace::eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (server_moe_trace *) user_data;

    int32_t il = -1;
    const int kind = trace_classify(ggml_get_name(t), il);
    const bool wanted = kind == TRACE_KIND_RESID ? (self->want_block_update && il <= self->n_layer)
                                                 : (kind != TRACE_KIND_NONE && il < self->n_layer);

    if (ask) {
        return wanted;
    }
    if (wanted) {
        if (kind == TRACE_KIND_RESID) {
            self->capture_resid(t, il);
        } else {
            self->capture(t, kind, il);
        }
    }
    return true; // continue the graph
}

void server_moe_trace::capture(struct ggml_tensor * t, int kind, int32_t il) {
    if (t->ne[2] != 1 || t->ne[3] != 1) {
        return;
    }
    const int64_t n0 = t->ne[0]; // n_expert_used (or 1 for the shared gate)
    const int64_t n1 = t->ne[1]; // n_tokens of this ubatch

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());

    auto at = [&](int64_t i, int64_t j) -> const uint8_t * {
        return buf.data() + i*t->nb[0] + j*t->nb[1];
    };
    auto to_f32 = [&](const uint8_t * p) -> float {
        switch (t->type) {
            case GGML_TYPE_F32:  return *(const float *) p;
            case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
            case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
            default:             return 0.0f;
        }
    };

    auto & L = layers[il];
    switch (kind) {
        case TRACE_KIND_EXPERTS:
            if (t->type != GGML_TYPE_I32) {
                return;
            }
            L.n_used_experts = (int32_t) n0;
            for (int64_t j = 0; j < n1; ++j) {
                for (int64_t i = 0; i < n0; ++i) {
                    L.experts.push_back(*(const int32_t *) at(i, j));
                }
            }
            L.n_tokens_experts += (int32_t) n1;
            break;
        case TRACE_KIND_WEIGHTS:
            L.n_used_weights = (int32_t) n0;
            for (int64_t j = 0; j < n1; ++j) {
                for (int64_t i = 0; i < n0; ++i) {
                    L.weights.push_back(to_f32(at(i, j)));
                }
            }
            L.n_tokens_weights += (int32_t) n1;
            break;
        case TRACE_KIND_SHARED:
            for (int64_t j = 0; j < n1; ++j) {
                L.shared.push_back(to_f32(at(0, j)));
            }
            L.n_tokens_shared += (int32_t) n1;
            break;
        default:
            break;
    }
}

// residual [n_embd, n_streams, n_tokens] (or [n_embd, n_tokens]); a long ubatch keeps only its last column, which is
// the one a token sampled after a prompt needs. The last layer may hold only the output rows (trailing tokens).
void server_moe_trace::capture_resid(struct ggml_tensor * t, int32_t idx) {
    if (t->ne[3] != 1 || idx < 0 || idx >= (int32_t) resid.size()) {
        return;
    }
    const bool    three = t->ne[2] != 1 || ggml_n_dims(t) == 3;
    const int64_t n0 = t->ne[0];
    const int64_t n1 = three ? t->ne[1] : 1;
    const int64_t nt = three ? t->ne[2] : t->ne[1];
    const size_t  nb_tok = three ? t->nb[2] : t->nb[1];
    if (idx == 0) {
        ubatch_base += ubatch_n_tok;
        ubatch_n_tok = (int32_t) nt;
    }
    const int32_t base = ubatch_base + (nt < ubatch_n_tok ? ubatch_n_tok - (int32_t) nt : 0);
    const int64_t t_first = nt > 16 ? nt - 1 : 0;

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    auto to_f32 = [&](const uint8_t * p) -> float {
        switch (t->type) {
            case GGML_TYPE_F32:  return *(const float *) p;
            case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
            case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
            default:             return 0.0f;
        }
    };
    auto & R = resid[idx];
    R.rows = n0*n1;
    for (int64_t tk = t_first; tk < nt; ++tk) {
        R.cols.push_back(base + (int32_t) tk);
        for (int64_t j = 0; j < n1; ++j) {
            for (int64_t i = 0; i < n0; ++i) {
                R.data.push_back(to_f32(buf.data() + i*t->nb[0] + j*(three ? t->nb[1] : 0) + tk*nb_tok));
            }
        }
    }
}

static const float * trace_resid_col(const server_moe_trace::resid_capture & R, int32_t col) {
    for (size_t k = 0; k < R.cols.size(); ++k) {
        if (R.cols[k] == col) {
            return R.data.data() + k*R.rows;
        }
    }
    return nullptr;
}

json server_moe_trace::token_json(int32_t col, llama_token tok, bool experts, bool block_update) const {
    json out = json {
        {"token_id", tok},
    };
    if (block_update) {
        json bu = json::array();
        for (int32_t il = 0; il < n_layer; ++il) {
            const auto & A = resid[il];
            const auto & B = resid[il + 1];
            const float * a = A.rows ? trace_resid_col(A, col) : nullptr;
            const float * b = B.rows == A.rows && A.rows ? trace_resid_col(B, col) : nullptr;
            if (!a || !b) {
                bu.push_back(nullptr);
                continue;
            }
            double num = 0.0, den = 0.0;
            for (int64_t i = 0; i < A.rows; ++i) {
                const double d = (double) b[i] - (double) a[i];
                num += d*d;
                den += (double) a[i]*(double) a[i];
            }
            bu.push_back(den > 0.0 ? (float) std::sqrt(num/den) : 0.0f);
        }
        out["block_update"]     = bu;
        out["block_update_def"] = "||l_last[il] - l_last[il-1]|| / ||l_last[il-1]|| (layer 0 against hc_init), L2 over all residual streams at this token's column";
    }
    if (!experts) {
        return out;
    }
    json moe = json::array();
    for (const auto & L : layers) {
        if (col < 0 || col >= L.n_tokens_experts) {
            moe.push_back(nullptr); // no router in this layer, or nothing captured for this column
            continue;
        }
        json experts = json::array();
        for (int32_t i = 0; i < L.n_used_experts; ++i) {
            experts.push_back(L.experts[(size_t) col*L.n_used_experts + i]);
        }
        json weights = json::array();
        if (col < L.n_tokens_weights) {
            for (int32_t i = 0; i < L.n_used_weights; ++i) {
                weights.push_back(L.weights[(size_t) col*L.n_used_weights + i]);
            }
        }
        json entry = json {
            {"experts", experts},
            {"weights", weights},
        };
        if (col < L.n_tokens_shared) {
            entry["shared"] = L.shared[col];
        }
        moe.push_back(entry);
    }
    out["moe"] = moe;
    return out;
}
