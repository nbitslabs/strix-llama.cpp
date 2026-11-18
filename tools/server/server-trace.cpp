#include "server-trace.h"

#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>

enum trace_kind {
    TRACE_KIND_NONE = 0,
    TRACE_KIND_EXPERTS,
    TRACE_KIND_WEIGHTS,
    TRACE_KIND_SHARED,
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
    return TRACE_KIND_NONE;
}

void server_moe_trace::init(int32_t n_layer_) {
    n_layer = n_layer_;
    layers.assign(n_layer, layer_capture());
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
}

bool server_moe_trace::eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (server_moe_trace *) user_data;

    int32_t il = -1;
    const int kind = trace_classify(ggml_get_name(t), il);
    const bool wanted = kind != TRACE_KIND_NONE && il < self->n_layer;

    if (ask) {
        return wanted;
    }
    if (wanted) {
        self->capture(t, kind, il);
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

json server_moe_trace::token_json(int32_t col, llama_token tok) const {
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
    return json {
        {"token_id", tok},
        {"moe",      moe},
    };
}
