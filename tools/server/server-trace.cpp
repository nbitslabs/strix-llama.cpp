#include "server-trace.h"

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <thread>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

// libllama's (unexported in llama.h) name -> weight map, the same one tests/test-quantize-stats reads
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

enum trace_kind {
    TRACE_KIND_NONE = 0,
    TRACE_KIND_EXPERTS,
    TRACE_KIND_WEIGHTS,
    TRACE_KIND_SHARED,
    TRACE_KIND_ATTENTION,
    TRACE_KIND_HEADIN,
    TRACE_KIND_RESID,
    TRACE_KIND_GDN_Q,
    TRACE_KIND_GDN_K,
    TRACE_KIND_GDN_V,
    TRACE_KIND_GDN_BETA,
    TRACE_KIND_GDN_GATE,
    TRACE_KIND_GDN_Z,
    TRACE_KIND_GDN_READ,
    TRACE_KIND_GDN_Y,
    TRACE_KIND_GDN_OUT,
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
    const char * names[] = {"q_conv_predelta", "k_conv_predelta", "v_conv_predelta", "beta_sigmoid", "gate", "z", "attn_output", "final_output", "linear_attn_out"};
    const char * copies[] = {"q_conv_trace", "k_conv_trace", "v_conv_trace", "beta_trace", "gate_trace", "z_trace", "attn_output_trace", "final_output_trace", "linear_attn_out_trace"};
    for (int i = 0; i < 9; ++i) {
        if (trace_match_layer(name, names[i], il) || trace_match_layer(name, copies[i], il)) {
            return TRACE_KIND_GDN_Q + i;
        }
    }
    if (trace_match_layer(name, "attn_weights", il)) {
        return TRACE_KIND_ATTENTION;
    }
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
    if (strcmp(name, "result_norm") == 0) {
        il = -1;
        return TRACE_KIND_HEADIN;
    }
    if (strcmp(name, "hc_init") == 0) {
        il = 0;
        return TRACE_KIND_RESID;
    }
    if (strcmp(name, "h_nextn") == 0) {
        il = -2; // the last layer's l_last, renamed when it is handed to the MTP head; mapped to n_layer below
        return TRACE_KIND_RESID;
    }
    return TRACE_KIND_NONE;
}

void server_moe_trace::init(int32_t n_layer_, int64_t n_embd_, int64_t n_stream_, const std::vector<bool> & recurrent) {
    gdn_layers = recurrent;
    n_layer  = n_layer_;
    n_embd   = n_embd_;
    n_stream = n_stream_ > 0 ? n_stream_ : 1;
    layers.assign(n_layer, layer_capture());
    attention.assign(n_layer, attention_capture());
    resid.assign(n_layer + 1, resid_capture());
}

void server_moe_trace::install(llama_context * ctx) {
    for (auto & l : gdn) {
        l.k.clear(); l.v.clear(); l.beta.clear(); l.a.clear(); l.present.clear();
        l.over_cap = false;
    }
    if (!installed) {
        llama_set_eval_callback(ctx, eval_cb, this);
        installed = true;
    }
}

void server_moe_trace::remove(llama_context * ctx) {
    llama_set_trace_attention(ctx, false);
    llama_set_trace_gdn(ctx, false);
    want_gdn = false;
    if (installed) {
        llama_set_eval_callback(ctx, nullptr, nullptr);
        installed = false;
    }
    begin_batch();
    lens.rows.clear(); // the dequantized output rows are per request; the mixer weights stay
}

void server_moe_trace::begin_batch(const std::vector<llama_pos> & positions) {
    begin_batch();
    col_pos = positions;
}

void server_moe_trace::begin_batch() {
    col_pos.clear();
    gdn_capture_us = 0;
    for (auto & l : gdn) {
        l.q.clear(); l.read.clear(); l.z.clear(); l.y.clear(); l.output.clear();
    }
    for (auto & l : layers) {
        l = layer_capture();
    }
    for (auto & r : resid) {
        r = resid_capture();
    }
    for (auto & a : attention) {
        a = attention_capture();
    }
    head_in = resid_capture();
    ubatch_base  = 0;
    ubatch_n_tok = 0;
}

bool server_moe_trace::eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (server_moe_trace *) user_data;

    int32_t il = -1;
    int kind = trace_classify(ggml_get_name(t), il);
    if (kind == TRACE_KIND_RESID && il == -2) {
        il = self->n_layer;
    }
    bool wanted;
    if (kind == TRACE_KIND_RESID) {
        wanted = (self->want_block_update || ((self->want_attention || self->want_gdn) && il == 0)) && il >= 0 && il <= self->n_layer &&
                 (self->resid_wanted.empty() || self->resid_wanted[il]);
    } else if (kind >= TRACE_KIND_GDN_Q && kind <= TRACE_KIND_GDN_OUT) {
        wanted = self->want_gdn && il >= 0 && il < (int32_t) self->gdn_layers.size() && self->gdn_layers[il];
    } else if (kind == TRACE_KIND_ATTENTION) {
        wanted = self->want_attention && il >= 0 && il < self->n_layer;
    } else if (kind == TRACE_KIND_HEADIN) {
        wanted = self->want_head_in;
    } else {
        wanted = kind != TRACE_KIND_NONE && il < self->n_layer;
    }

    if (ask) {
        return wanted;
    }
    if (wanted) {
        if (kind == TRACE_KIND_RESID) {
            self->capture_resid(t, il);
        } else if (kind >= TRACE_KIND_GDN_Q && kind <= TRACE_KIND_GDN_OUT) {
            self->capture_gdn(t, kind, il);
        } else if (kind == TRACE_KIND_ATTENTION) {
            self->capture_attention(t, il);
        } else if (kind == TRACE_KIND_HEADIN) {
            self->capture_head_in(t);
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

// residual [n_embd, n_streams, n_tokens] (or flat [n_embd*n_streams, n_tokens]); the stream count comes from the model,
// since a one-token [n_embd, n_streams, 1] has the same dims as [n_embd, n_tokens]. A long ubatch keeps only its last
// column, which is the one a token sampled after a prompt needs. The last layer may hold only the output rows.
void server_moe_trace::capture_resid(struct ggml_tensor * t, int32_t idx) {
    if (t->ne[3] != 1 || idx < 0 || idx >= (int32_t) resid.size()) {
        return;
    }
    const bool three = t->ne[0] == n_embd && t->ne[1] == n_stream;
    if (!three && t->ne[0] != n_embd*n_stream) {
        return; // not a wide residual of this model
    }
    const int64_t n0 = three ? t->ne[0] : n_embd;
    const int64_t n1 = n_stream;
    const int64_t nt = three ? t->ne[2] : t->ne[1];
    const size_t  nb_str = three ? t->nb[1] : n_embd*t->nb[0];
    const size_t  nb_tok = three ? t->nb[2] : t->nb[1];
    if (idx == 0) {
        ubatch_base += ubatch_n_tok;
        ubatch_n_tok = (int32_t) nt;
    }
    const int32_t base = ubatch_base + (nt < ubatch_n_tok ? ubatch_n_tok - (int32_t) nt : 0);
    const int64_t t_first = nt > 16 ? nt - 1 : 0;
    auto & R = resid[idx];
    for (int32_t c : R.cols) {
        if (c == base + (int32_t) t_first) {
            return; // already captured (l_last and h_nextn can both name the last layer's residual)
        }
    }

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
    R.rows = n0*n1;
    for (int64_t tk = t_first; tk < nt; ++tk) {
        R.cols.push_back(base + (int32_t) tk);
        for (int64_t j = 0; j < n1; ++j) {
            for (int64_t i = 0; i < n0; ++i) {
                R.data.push_back(to_f32(buf.data() + i*t->nb[0] + j*nb_str + tk*nb_tok));
            }
        }
    }
}

void server_moe_trace::resid_filter(const std::vector<int32_t> * blocks) {
    resid_wanted.clear();
    if (!blocks) {
        return;
    }
    resid_wanted.assign(resid.size(), 0);
    resid_wanted[0] = 1;
    for (int32_t il : *blocks) {
        if (il + 1 >= 0 && il + 1 < (int32_t) resid.size()) {
            resid_wanted[il + 1] = 1;
        }
    }
}

// the head input [n_embd, n_tokens] (result_norm): same column bookkeeping as the residuals, which hc_init maintains
void server_moe_trace::capture_head_in(struct ggml_tensor * t) {
    if (t->ne[0] != n_embd || t->ne[2] != 1 || t->ne[3] != 1) {
        return;
    }
    const int64_t nt = t->ne[1];
    const int32_t base = ubatch_base + (nt < ubatch_n_tok ? ubatch_n_tok - (int32_t) nt : 0);
    const int64_t t_first = nt > 16 ? nt - 1 : 0;
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    head_in.rows = n_embd;
    for (int64_t tk = t_first; tk < nt; ++tk) {
        head_in.cols.push_back(base + (int32_t) tk);
        for (int64_t i = 0; i < n_embd; ++i) {
            const uint8_t * p = buf.data() + i*t->nb[0] + tk*t->nb[1];
            float v = 0.0f;
            switch (t->type) {
                case GGML_TYPE_F32:  v = *(const float *) p; break;
                case GGML_TYPE_F16:  v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p); break;
                case GGML_TYPE_BF16: v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) p); break;
                default: break;
            }
            head_in.data.push_back(v);
        }
    }
}

void server_moe_trace::capture_attention(struct ggml_tensor * t, int32_t il) {
    if (t->type != GGML_TYPE_F32 || t->ne[3] != 1) {
        return;
    }
    auto & a = attention[il];
    a.n_kv = t->ne[0];
    a.n_head = t->ne[2];
    const int64_t nt = t->ne[1];
    const int32_t base = ubatch_base + ubatch_n_tok - (int32_t) nt;
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    for (int64_t j = 0; j < nt; ++j) {
        a.cols.push_back(base + (int32_t) j);
        a.data.emplace_back();
        a.data.back().reserve(a.n_head*a.n_kv);
        for (int64_t h = 0; h < a.n_head; ++h) {
            for (int64_t p = 0; p < a.n_kv; ++p) {
                float v;
                memcpy(&v, buf.data() + p*t->nb[0] + j*t->nb[1] + h*t->nb[2], sizeof(v));
                a.data.back().push_back(v);
            }
        }
    }
}

json server_moe_trace::attention_json(int32_t col, int64_t n_pos) const {
    json out = {{"blocks", json::array()}, {"heads", json::array()}};
    for (int32_t il = 0; il < n_layer; ++il) {
        const auto & a = attention[il];
        if (!a.n_head) {
            continue;
        }
        out["blocks"].push_back(il);
        auto it = std::find(a.cols.begin(), a.cols.end(), col);
        if (it == a.cols.end() || n_pos < 0) {
            out["heads"].push_back(nullptr);
            continue;
        }
        const auto & data = a.data[it - a.cols.begin()];
        const int64_t n_kv = data.size()/a.n_head;
        if (n_pos > n_kv) {
            out["heads"].push_back(nullptr);
            continue;
        }
        json heads = json::array();
        for (int64_t h = 0; h < a.n_head; ++h) {
            const float * row = data.data() + h*n_kv;
            heads.push_back(std::vector<float>(row, row + n_pos));
        }
        out["heads"].push_back(std::move(heads));
    }
    return out;
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

// a whole weight (or rows [r0, r0 + nr) of it) as f32, through the type's dequantizer
static bool trace_tensor_f32(const ggml_tensor * t, int64_t r0, int64_t nr, std::vector<float> & out) {
    if (t == nullptr || !ggml_is_contiguous(t) || r0 < 0 || r0 + nr > ggml_nrows(t)) {
        return false;
    }
    const auto * tt = ggml_get_type_traits(t->type);
    if (t->type != GGML_TYPE_F32 && tt->to_float == nullptr) {
        return false;
    }
    const int64_t ne0 = t->ne[0];
    std::vector<uint8_t> buf(t->nb[1]*nr);
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), buf.data(), t->nb[1]*r0, buf.size());
    out.resize(ne0*nr);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), buf.data(), out.size()*sizeof(float));
    } else {
        for (int64_t r = 0; r < nr; ++r) {
            tt->to_float(buf.data() + r*t->nb[1], out.data() + r*ne0, ne0);
        }
    }
    return true;
}

bool server_moe_trace::lens_prepare(const llama_model * model, std::string & err, bool checkpoints) {
    if (lens.ready) {
        return !checkpoints || lens_prepare_head(err);
    }
    auto find = [&](const char * name) -> ggml_tensor * {
        for (const auto & [n, t] : llama_internal_get_tensor_map(model)) {
            if (n == name) {
                return t;
            }
        }
        return nullptr;
    };
    ggml_tensor * norm = find("output_hc_norm.weight");
    ggml_tensor * down = find("output_hc_down.weight");
    ggml_tensor * up   = find("output_hc_up.weight");
    ggml_tensor * out  = find("output.weight");
    if (out == nullptr) {
        out = find("token_embd.weight"); // tied head
    }
    if (!norm || !down || !up || !out) {
        err = "block_push needs a hyper-connection model with a final hc mixer (output_hc_norm/down/up) and an output head";
        return false;
    }
    lens_weights L;
    L.n_embd = llama_model_n_embd(model);
    L.hc     = ggml_nelements(norm) / L.n_embd;
    L.hc_lr  = down->ne[1];
    if (L.hc < 1 || down->ne[0] != L.hc*L.n_embd || up->ne[0] != L.hc_lr || up->ne[1] != L.hc*L.n_embd || out->ne[0] != L.n_embd) {
        err = "block_push: unexpected shapes of the final hc mixer or output head";
        return false;
    }
    if (!trace_tensor_f32(norm, 0, ggml_nrows(norm), L.norm) || !trace_tensor_f32(down, 0, down->ne[1], L.down) ||
        !trace_tensor_f32(up, 0, up->ne[1], L.up)) {
        err = "block_push: cannot dequantize the final hc mixer weights";
        return false;
    }
    if (ggml_tensor * os = find("output.scale"); os && ggml_nelements(os) == 1) {
        std::vector<float> v;
        if (trace_tensor_f32(os, 0, 1, v)) {
            L.out_scale = v[0];
        }
    }
    char arch[64] = {0};
    char buf[64]  = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) > 0 &&
        llama_model_meta_val_str(model, (std::string(arch) + ".attention.layer_norm_rms_epsilon").c_str(), buf, sizeof(buf)) > 0) {
        L.eps = strtof(buf, nullptr);
    }
    L.output = out;
    L.ready  = true;
    lens = std::move(L);
    return !checkpoints || lens_prepare_head(err);
}

const std::vector<float> & server_moe_trace::lens_row(llama_token tok) {
    auto it = lens.rows.find(tok);
    if (it == lens.rows.end()) {
        std::vector<float> row;
        if (!trace_tensor_f32(lens.output, tok, 1, row)) {
            row.assign(lens.n_embd, 0.0f);
        }
        it = lens.rows.emplace(tok, std::move(row)).first;
    }
    return it->second;
}

// same math as build_hc_mix(x, hc_head_norm, hc_head_down, hc_head_up) with no inject (the unfused f32 path):
// per-stream RMSNorm * gamma, gate = up(silu(down(xn)/hc)), out = mean over streams of xn * sigmoid(gate)
void server_moe_trace::lens_hidden(const float * x, std::vector<float> & out) const {
    const int64_t n = lens.n_embd, hc = lens.hc, hd = hc*n, lr = lens.hc_lr;
    std::vector<float> xn(hd);
    for (int64_t j = 0; j < hc; ++j) {
        double ss = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            ss += (double) x[j*n + i]*x[j*n + i];
        }
        const float scale = 1.0f/std::sqrt((float) (ss/n) + lens.eps);
        for (int64_t i = 0; i < n; ++i) {
            xn[j*n + i] = x[j*n + i]*scale*lens.norm[j*n + i];
        }
    }
    std::vector<float> lo(lr);
    for (int64_t r = 0; r < lr; ++r) {
        double acc = 0.0;
        const float * w = lens.down.data() + r*hd;
        for (int64_t k = 0; k < hd; ++k) {
            acc += (double) w[k]*xn[k];
        }
        const float v = (float) acc/(float) hc;
        lo[r] = v/(1.0f + std::exp(-v));
    }
    out.assign(n, 0.0f);
    for (int64_t k = 0; k < hd; ++k) {
        double acc = 0.0;
        const float * w = lens.up.data() + k*lr;
        for (int64_t r = 0; r < lr; ++r) {
            acc += (double) w[r]*lo[r];
        }
        const float g = 1.0f/(1.0f + std::exp(-(float) acc));
        out[k % n] += xn[k]*g;
    }
    for (auto & v : out) {
        v /= (float) hc;
    }
}

struct server_trace_lens_mix {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_cgraph * graph = nullptr;

    ~server_trace_lens_mix() {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
};

bool server_moe_trace::lens_prepare_mix() {
    if (lens.mix) {
        return true;
    }
    const int64_t n = lens.n_embd, hc = lens.hc, lr = lens.hc_lr;
    if (n <= 0 || hc <= 0 || lr <= 0 || n > INT64_MAX/hc) {
        return false;
    }
    const int64_t hd = n*hc;
    if (hd > INT64_MAX/lr || lens.norm.size() != (size_t) hd ||
        lens.down.size() != (size_t) (hd*lr) || lens.up.size() != (size_t) (hd*lr)) {
        return false;
    }
    auto M = std::make_shared<server_trace_lens_mix>();
    M->backend = ggml_backend_cpu_init();
    M->ctx = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true});
    if (!M->backend || !M->ctx) {
        return false;
    }
    ggml_backend_cpu_set_n_threads(M->backend, (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
    const int64_t nc = std::max<int64_t>(1, (int64_t) n_layer + 1);
    auto * ctx = M->ctx;
    M->norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, hc);
    M->down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, lr);
    M->up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lr, hd);
    M->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, nc);
    auto * xn = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, M->input, n, hc, nc), lens.eps), M->norm);
    xn = ggml_reshape_2d(ctx, xn, hd, nc);
    auto * lo = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, M->down, xn), 1.0f/hc));
    auto * g = ggml_sigmoid(ctx, ggml_mul_mat(ctx, M->up, lo));
    auto * y = ggml_reshape_3d(ctx, ggml_mul(ctx, xn, g), n, hc, nc);
    y = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3)));
    M->hidden = ggml_scale(ctx, ggml_reshape_2d(ctx, y, n, nc), 1.0f/hc);
    M->graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(M->graph, M->hidden);
    M->buffer = ggml_backend_alloc_ctx_tensors(ctx, M->backend);
    if (!M->buffer) {
        return false;
    }
    ggml_backend_tensor_set(M->norm, lens.norm.data(), 0, ggml_nbytes(M->norm));
    ggml_backend_tensor_set(M->down, lens.down.data(), 0, ggml_nbytes(M->down));
    ggml_backend_tensor_set(M->up, lens.up.data(), 0, ggml_nbytes(M->up));
    lens.mix = std::move(M);
    return true;
}

void server_moe_trace::lens_hidden_batch(const std::vector<const float *> & xs, std::vector<float> & out) {
    out.resize(xs.size()*lens.n_embd);
    if (xs.empty()) {
        return;
    }
    if (lens_prepare_mix()) {
        auto & M = *lens.mix;
        const size_t hd = M.input->ne[0], nc = M.input->ne[1];
        // Nonzero padding avoids 0/0 in RMSNorm when epsilon is zero.
        std::vector<float> input(hd*nc, lens.eps == 0.0f ? 1.0f : 0.0f);
        bool success = true;
        for (size_t start = 0; start < xs.size(); start += nc) {
            const size_t count = std::min(nc, xs.size() - start);
            std::fill(input.begin(), input.end(), lens.eps == 0.0f ? 1.0f : 0.0f);
            for (size_t c = 0; c < count; ++c) {
                std::copy_n(xs[start + c], hd, input.data() + c*hd);
            }
            ggml_backend_tensor_set(M.input, input.data(), 0, ggml_nbytes(M.input));
            if (ggml_backend_graph_compute(M.backend, M.graph) != GGML_STATUS_SUCCESS) {
                success = false;
                break;
            }
            ggml_backend_tensor_get(M.hidden, out.data() + start*lens.n_embd, 0, count*lens.n_embd*sizeof(float));
        }
        if (success) {
            return;
        }
    }
    std::vector<float> h;
    for (size_t c = 0; c < xs.size(); ++c) {
        lens_hidden(xs[c], h);
        std::copy(h.begin(), h.end(), out.data() + c*lens.n_embd);
    }
}

json server_moe_trace::block_push_json(int32_t col, const std::vector<llama_token> & tokens, const std::vector<float> & logits) {
    json ids = json::array();
    for (llama_token t : tokens) {
        ids.push_back(t);
    }
    json out = json {
        {"token_ids", ids},
    };
    if (!lens.ready) {
        out["error"] = "lens weights not loaded";
        return out;
    }
    // lens logit per residual (hc_init, then after every block), per token; null where a residual is missing
    std::vector<std::vector<double>> lg(n_layer + 1);
    std::vector<bool> ok(n_layer + 1, false);
    std::vector<const float *> xs;
    std::vector<int32_t> residuals;
    for (int32_t r = 0; r <= n_layer; ++r) {
        const auto & R = resid[r];
        const float * x = R.rows == lens.hc*lens.n_embd ? trace_resid_col(R, col) : nullptr;
        if (!x) {
            continue;
        }
        xs.push_back(x);
        residuals.push_back(r);
    }
    std::vector<float> hidden;
    lens_hidden_batch(xs, hidden);
    for (size_t c = 0; c < xs.size(); ++c) {
        const int32_t r = residuals[c];
        const float * h = hidden.data() + c*lens.n_embd;
        lg[r].resize(tokens.size());
        for (size_t t = 0; t < tokens.size(); ++t) {
            const auto & w = lens_row(tokens[t]);
            double acc = 0.0;
            for (int64_t i = 0; i < lens.n_embd; ++i) {
                acc += (double) w[i]*h[i];
            }
            lg[r][t] = acc*lens.out_scale;
        }
        ok[r] = true;
    }
    json base = json::array();
    json push = json::array();
    json fin  = json::array();
    for (size_t t = 0; t < tokens.size(); ++t) {
        base.push_back(ok[0] ? json((float) lg[0][t]) : json(nullptr));
        json p = json::array();
        for (int32_t il = 0; il < n_layer; ++il) {
            p.push_back(ok[il] && ok[il + 1] ? json((float) (lg[il + 1][t] - lg[il][t])) : json(nullptr));
        }
        push.push_back(p);
        fin.push_back(t < logits.size() && std::isfinite(logits[t]) ? json(logits[t]) : json(nullptr));
    }
    out["base"]  = base;
    out["push"]  = push;
    out["logit"] = fin;
    return out;
}

// Keep the quantized head and graph alive with the lens; no vocabulary-sized f32 weight cache is needed.
struct server_trace_head {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * output = nullptr; // the model's tensor (device path) or a host copy owned by ctx/buffer (CPU path)
    bool owns_output = false;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_cgraph * graph = nullptr;

    ~server_trace_head() {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
};

bool server_moe_trace::lens_prepare_head(std::string & err) {
    if (lens.head) {
        return true;
    }
    auto H = std::make_shared<server_trace_head>();
    auto * out = lens.output;
    if (!ggml_is_contiguous(out) || out->ne[2] != 1 || out->ne[3] != 1 ||
        (out->type != GGML_TYPE_F32 && !ggml_get_type_traits(out->type)->to_float)) {
        err = "unsupported output head layout or type";
        return false;
    }
    // Run the head on the device that holds the model's output tensor, with the model's own tensor and the backend's
    // own mat-vec kernel: no host copy, and the same activation quantization the engine uses for its final logits,
    // so the block-after-the-last-layer row reproduces the engine's distribution. A CPU backend with a host copy is
    // the fallback (models on the CPU, or a device whose backend cannot be instantiated).
    ggml_backend_dev_t dev = out->buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(out->buffer)) : nullptr;
    if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        H->backend = ggml_backend_dev_init(dev, nullptr);
    }
    const bool on_device = H->backend != nullptr;
    if (!on_device) {
        H->backend = ggml_backend_cpu_init();
        if (H->backend) {
            ggml_backend_cpu_set_n_threads(H->backend, (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
        }
    }
    H->ctx = ggml_init({4*ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true});
    if (!H->backend || !H->ctx) {
        err = "cannot allocate the head's backend or graph context";
        return false;
    }
    // the device path uses the model tensor itself; the CPU path works on a same-type host copy
    H->output = on_device ? out : ggml_dup_tensor(H->ctx, out);
    H->owns_output = !on_device;
    if (on_device || ggml_get_type_traits_cpu(out->type)->vec_dot) {
        H->hidden = ggml_new_tensor_2d(H->ctx, GGML_TYPE_F32, lens.n_embd, 1);
        H->logits = ggml_mul_mat(H->ctx, H->output, H->hidden);
        H->graph = ggml_new_graph_custom(H->ctx, 8, false);
        ggml_build_forward_expand(H->graph, H->logits);
    }
    H->buffer = ggml_backend_alloc_ctx_tensors(H->ctx, H->backend);
    if (!H->buffer) {
        err = "cannot allocate the head's buffers";
        return false;
    }
    if (H->owns_output) {
        ggml_backend_tensor_get(out, H->output->data, 0, ggml_nbytes(out));
    }
    lens.head = std::move(H);
    return true;
}

json server_moe_trace::checkpoints_json(int32_t col, llama_token tok, const std::vector<int32_t> & blocks, int32_t top_k) {
    json out = {{"blocks", blocks}, {"top", json::array()}, {"final_prob", json::array()},
                {"final_rank", json::array()}, {"entropy", json::array()}};
    GGML_ASSERT(lens.head);
    auto & H = *lens.head;
    const int64_t nv = H.output->ne[1];
    GGML_ASSERT(tok >= 0 && tok < nv);
    std::vector<float> h, logits(nv), rows;
    std::vector<int32_t> ids(nv);
    const int64_t nk = std::min<int64_t>(top_k, nv);
    std::vector<const float *> xs;
    for (int32_t il : blocks) {
        const auto & R = resid.at(il + 1);
        const float * x = R.rows == lens.hc*lens.n_embd ? trace_resid_col(R, col) : nullptr;
        const float * hin = il == n_layer - 1 && head_in.rows == lens.n_embd ? trace_resid_col(head_in, col) : nullptr;
        if (x && !hin) {
            xs.push_back(x);
        }
    }
    std::vector<float> hidden;
    lens_hidden_batch(xs, hidden);
    size_t mixed = 0;
    for (int32_t il : blocks) {
        const auto & R = resid.at(il + 1);
        const float * x = R.rows == lens.hc*lens.n_embd ? trace_resid_col(R, col) : nullptr;
        if (!x) {
            for (const char * key : {"top", "final_prob", "final_rank", "entropy"}) {
                out[key].push_back(nullptr);
            }
            continue;
        }
        // the last block's row takes the engine's own head input when it was captured, so it reproduces the engine
        const float * hin = il == n_layer - 1 && head_in.rows == lens.n_embd ? trace_resid_col(head_in, col) : nullptr;
        if (hin) {
            h.assign(hin, hin + lens.n_embd);
        } else {
            const float * mix = hidden.data() + mixed++*lens.n_embd;
            h.assign(mix, mix + lens.n_embd);
        }
        if (H.graph) {
            ggml_backend_tensor_set(H.hidden, h.data(), 0, h.size()*sizeof(float));
            if (ggml_backend_graph_compute(H.backend, H.graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("checkpoints: CPU output head computation failed");
            }
            ggml_backend_tensor_get(H.logits, logits.data(), 0, logits.size()*sizeof(float));
        } else {
            // Types without a CPU vec_dot use bounded row chunks, so fallback never expands the whole head to f32.
            for (int64_t r0 = 0; r0 < nv; r0 += 32) {
                const int64_t nr = std::min<int64_t>(32, nv - r0);
                if (!trace_tensor_f32(H.output, r0, nr, rows)) {
                    throw std::runtime_error("checkpoints: cannot dequantize output rows");
                }
                for (int64_t r = 0; r < nr; ++r) {
                    double acc = 0.0;
                    for (int64_t i = 0; i < lens.n_embd; ++i) {
                        acc += (double) rows[r*lens.n_embd + i]*h[i];
                    }
                    logits[r0 + r] = (float) acc;
                }
            }
        }
        for (float & v : logits) {
            v *= lens.out_scale;
        }
        const double max_logit = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0, weighted = 0.0;
        int64_t rank = 1;
        for (float v : logits) {
            const double d = (double) v - max_logit;
            const double e = std::exp(d);
            sum += e;
            weighted += e*d;
            rank += v > logits[tok];
        }
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + nk, ids.end(), [&](int32_t a, int32_t b) {
            return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
        });
        json top = json::array();
        for (int64_t k = 0; k < nk; ++k) {
            top.push_back({{"id", ids[k]}, {"prob", std::exp((double) logits[ids[k]] - max_logit)/sum}});
        }
        out["top"].push_back(std::move(top));
        out["final_prob"].push_back(std::exp((double) logits[tok] - max_logit)/sum);
        out["final_rank"].push_back(rank);
        out["entropy"].push_back(std::log(sum) - weighted/sum);
    }
    return out;
}

std::vector<float> server_trace_gdn_sweep(int s, int h, int kh, int n, const ggml_fp16_t * k, const float * beta, const float * a, const float * q) {
    std::vector<float> weights(n*h);
    const float scale = 1.0f/std::sqrt((float) s);
    for (int head = 0; head < h; ++head) {
        std::vector<float> r(q + (head%kh)*s, q + (head%kh + 1)*s);
        for (int i = n - 1; i >= 0; --i) {
            const auto * key = k + (i*kh + head%kh)*s;
            float dot = 0;
            for (int d = 0; d < s; ++d) {
                dot += ggml_fp16_to_fp32(key[d])*r[d];
            }
            weights[i*h + head] = beta[i*h + head]*dot*scale;
            for (int d = 0; d < s; ++d) {
                r[d] = a[i*h + head]*(r[d] - beta[i*h + head]*ggml_fp16_to_fp32(key[d])*dot);
            }
        }
    }
    return weights;
}

std::vector<float> server_trace_gdn_forward(int s, int h, int kh, int n, const ggml_fp16_t * k, const ggml_fp16_t * v, const float * beta, const float * a, const float * q) {
    std::vector<float> out(s*h, 0);
    for (int head = 0; head < h; ++head) {
        std::vector<float> state(s*s, 0);
        for (int i = 0; i < n; ++i) {
            const auto * key = k + (i*kh + head%kh)*s;
            for (float & x : state) { x *= a[i*h + head]; }
            for (int d = 0; d < s; ++d) {
                float dot = 0;
                for (int j = 0; j < s; ++j) { dot += state[d*s + j]*ggml_fp16_to_fp32(key[j]); }
                const float delta = beta[i*h + head]*(ggml_fp16_to_fp32(v[(i*h + head)*s + d]) - dot);
                for (int j = 0; j < s; ++j) { state[d*s + j] += ggml_fp16_to_fp32(key[j])*delta; }
            }
        }
        for (int d = 0; d < s; ++d) {
            for (int j = 0; j < s; ++j) { out[head*s + d] += state[d*s + j]*q[(head%kh)*s + j]; }
            out[head*s + d] /= std::sqrt((float) s);
        }
    }
    return out;
}

std::vector<float> server_trace_gdn_norm(int s, int h, const float * read, const float * norm, bool shared, const float * z, float eps) {
    std::vector<float> factors(s*h);
    for (int head = 0; head < h; ++head) {
        float ss = 0;
        for (int d = 0; d < s; ++d) { ss += read[head*s + d]*read[head*s + d]; }
        const float rms = std::sqrt(ss/s + eps);
        for (int d = 0; d < s; ++d) {
            const int j = head*s + d;
            // qwen4exp uses sigmoid here; SiLU belongs to the preceding convolution.
            factors[j] = norm[shared ? d : j]/((1.0f + std::exp(-z[j]))*rms);
        }
    }
    return factors;
}

void server_trace_gdn_shares(int s, int h, int n, int out_dim, const ggml_fp16_t * v, const float * weights, const float * read, const float * factors, const ggml_fp16_t * out_weight, const float * output, std::vector<double> & shares, std::vector<double> & heads) {
    const int width = s*h;
    // g = W_out^T O: the one pass over the (fp16) projection weight per token; rows are converted in bulk, a scalar
    // conversion per element was the hot spot (out_dim*width calls per block)
    std::vector<float> g(width, 0), wrow(width), vrow(width);
    double out_norm = 0;
    for (int row = 0; row < out_dim; ++row) {
        out_norm += (double) output[row]*output[row];
        ggml_fp16_to_fp32_row(out_weight + (size_t) row*width, wrow.data(), width);
        const float o = output[row];
        for (int j = 0; j < width; ++j) { g[j] += wrow[j]*o; }
    }
    std::vector<double> read_norm(h, 0);
    for (int head = 0; head < h; ++head) {
        for (int d = 0; d < s; ++d) { read_norm[head] += (double) read[head*s + d]*read[head*s + d]; }
    }
    shares.assign(n, 0);
    heads.assign(n*h, 0);
    for (int i = 0; i < n; ++i) {
        ggml_fp16_to_fp32_row(v + (size_t) i*width, vrow.data(), width);
        double projected = 0;
        for (int head = 0; head < h; ++head) {
            const float w = weights[i*h + head];
            double dot = 0, proj = 0;
            for (int d = 0; d < s; ++d) {
                const int j = head*s + d;
                const double c = (double) w*vrow[j];
                dot += c*read[j];
                proj += c*factors[j]*g[j];
            }
            heads[i*h + head] = read_norm[head] > 0 ? dot/read_norm[head] : 0;
            projected += proj;
        }
        shares[i] = out_norm > 0 ? projected/out_norm : 0;
    }
}

// The GGUF shards of the loaded model, metadata only. The engine repacks dense q8_0 weights in place on the GPU (MMB
// "dense R2"), and reading such a tensor back aborts, so the GDN attribution takes W_out from the files: same bytes the
// engine loaded, dequantized once on the CPU.
struct trace_gguf_shards {
    struct shard { std::string path; gguf_context * meta = nullptr; };
    std::vector<shard> shards;

    ~trace_gguf_shards() {
        for (auto & s : shards) {
            if (s.meta) { gguf_free(s.meta); }
        }
    }

    bool open(const std::string & model_path, std::string & err) {
        gguf_init_params p = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * first = gguf_init_from_file(model_path.c_str(), p);
        if (!first) { err = "gdn: cannot read the model file " + model_path; return false; }
        shards.push_back({model_path, first});
        int split_count = 1, split_no = 0;
        if (int64_t k = gguf_find_key(first, "split.count"); k >= 0) { split_count = gguf_get_val_u16(first, k); }
        if (int64_t k = gguf_find_key(first, "split.no"); k >= 0) { split_no = gguf_get_val_u16(first, k); }
        if (split_count <= 1) { return true; }
        char prefix[4096], path[4096];
        if (!llama_split_prefix(prefix, sizeof(prefix), model_path.c_str(), split_no, split_count)) {
            err = "gdn: cannot derive the shard names of " + model_path;
            return false;
        }
        for (int i = 0; i < split_count; ++i) {
            if (i == split_no) { continue; }
            llama_split_path(path, sizeof(path), prefix, i, split_count);
            gguf_context * g = gguf_init_from_file(path, p);
            if (!g) { err = std::string("gdn: cannot read the model shard ") + path; return false; }
            shards.push_back({path, g});
        }
        return true;
    }

    // the named [ne0, ne1] tensor as fp16 rows, scaled; dequantized through the on-disk type's traits
    bool read_f16(const std::string & name, int64_t ne0, int64_t ne1, float scale, std::vector<ggml_fp16_t> & out, std::string & err) {
        for (auto & s : shards) {
            const int64_t id = gguf_find_tensor(s.meta, name.c_str());
            if (id < 0) { continue; }
            const ggml_type type = gguf_get_tensor_type(s.meta, id);
            const auto * tt = ggml_get_type_traits(type);
            const size_t row_size = ggml_row_size(type, ne0);
            if (gguf_get_tensor_size(s.meta, id) != row_size*ne1 || (type != GGML_TYPE_F32 && tt->to_float == nullptr)) {
                err = "gdn: " + name + " has an unexpected shape or type in " + s.path;
                return false;
            }
            std::vector<uint8_t> buf(row_size*ne1);
            FILE * f = fopen(s.path.c_str(), "rb");
            const size_t off = gguf_get_data_offset(s.meta) + gguf_get_tensor_offset(s.meta, id);
            const bool ok = f && fseeko(f, (off_t) off, SEEK_SET) == 0 && fread(buf.data(), 1, buf.size(), f) == buf.size();
            if (f) { fclose(f); }
            if (!ok) { err = "gdn: cannot read " + name + " from " + s.path; return false; }
            out.resize(ne0*ne1);
            std::vector<float> row(ne0);
            for (int64_t r = 0; r < ne1; ++r) {
                if (type == GGML_TYPE_F32) {
                    memcpy(row.data(), buf.data() + r*row_size, ne0*sizeof(float));
                } else {
                    tt->to_float(buf.data() + r*row_size, row.data(), ne0);
                }
                if (scale != 1.0f) {
                    for (float & x : row) { x *= scale; }
                }
                ggml_fp32_to_fp16_row(row.data(), out.data() + r*ne0, ne0);
            }
            return true;
        }
        err = "gdn: " + name + " is not in the model files";
        return false;
    }
};

bool server_moe_trace::gdn_prepare(const llama_model * model, const std::string & model_path, std::string & err) {
    if (!gdn.empty()) { return true; }
    trace_gguf_shards files;
    if (!files.open(model_path, err)) { return false; }
    char arch[128] = {}, text[128] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "qwen4exp") != 0) {
        err = "gdn: only qwen4exp is supported";
        return false;
    }
    auto find = [&](const std::string & name) -> ggml_tensor * {
        for (const auto & item : llama_internal_get_tensor_map(model)) {
            if (item.first == name) { return item.second; }
        }
        return nullptr;
    };
    auto meta_int = [&](const char * suffix) {
        if (llama_model_meta_val_str(model, (std::string(arch) + suffix).c_str(), text, sizeof(text)) <= 0) { return 0; }
        return atoi(text);
    };
    const int s = meta_int(".ssm.state_size"), h = meta_int(".ssm.time_step_rank"), kh = meta_int(".ssm.group_count");
    if (s <= 0 || h <= 0 || kh <= 0 || h%kh != 0 || s > 4096 || h > 256 || (int64_t) s*h*16384 > INT32_MAX) {
        err = "gdn: invalid state/head dimensions";
        return false;
    }
    if (llama_model_meta_val_str(model, (std::string(arch) + ".attention.layer_norm_rms_epsilon").c_str(), text, sizeof(text)) <= 0) {
        err = "gdn: missing RMS epsilon";
        return false;
    }
    const float eps = strtof(text, nullptr);
    if (!std::isfinite(eps) || eps <= 0) { err = "gdn: invalid RMS epsilon"; return false; }
    std::vector<gdn_layer> prepared;
    for (int il = 0; il < llama_model_n_layer(model); ++il) {
        if (!llama_model_layer_is_recurrent(model, il)) { continue; }
        gdn_layer l;
        l.il = il; l.s = s; l.h = h; l.kh = kh; l.eps = eps;
        const std::string prefix = "blk." + std::to_string(il);
        auto * norm = find(prefix + ".ssm_norm.weight");
        auto * out = find(prefix + ".ssm_out.weight");
        if (!norm || !out || (ggml_nelements(norm) != s && ggml_nelements(norm) != s*h) || out->ne[0] != s*h ||
                out->ne[1] != llama_model_n_embd(model) || !trace_tensor_f32(norm, 0, ggml_nrows(norm), l.norm)) {
            err = "gdn: unavailable or unsupported norm/output weights at block " + std::to_string(il);
            return false;
        }
        if (out->ne[1] <= 0 || out->ne[1] > INT32_MAX/(s*h)) {
            err = "gdn: output dimensions are too large";
            return false;
        }
        l.out_dim = out->ne[1];
        float out_scale = 1.0f;
        if (auto * scale = find(prefix + ".ssm_out.scale")) {
            std::vector<float> scalar;
            if (ggml_nelements(scale) != 1 || !trace_tensor_f32(scale, 0, 1, scalar)) {
                err = "gdn: unsupported output scale";
                return false;
            }
            out_scale = scalar[0];
        }
        if (!files.read_f16(prefix + ".ssm_out.weight", s*h, l.out_dim, out_scale, l.out_weight, err)) {
            return false;
        }
        prepared.push_back(std::move(l));
    }
    if (prepared.empty()) { err = "gdn: model has no recurrent blocks"; return false; }
    gdn = std::move(prepared);
    return true;
}

void server_moe_trace::capture_gdn(ggml_tensor * t, int kind, int32_t il) {
    auto it = std::find_if(gdn.begin(), gdn.end(), [&](const gdn_layer & l) { return l.il == il; });
    if (it == gdn.end() || t->ne[3] != 1 || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) { return; }
    auto & l = *it;
    const bool headed = kind <= TRACE_KIND_GDN_BETA || kind == TRACE_KIND_GDN_READ;
    const int64_t nt = headed ? t->ne[2] : t->ne[1];
    const int64_t width = headed ? t->ne[0]*t->ne[1] : t->ne[0];
    const int expected = kind == TRACE_KIND_GDN_Q || kind == TRACE_KIND_GDN_K ? l.s*l.kh :
        kind == TRACE_KIND_GDN_BETA || kind == TRACE_KIND_GDN_GATE ? l.h : kind == TRACE_KIND_GDN_OUT ? l.out_dim : l.s*l.h;
    if (width != expected || (!headed && t->ne[2] != 1)) { return; }
    const int32_t base = ubatch_base + ubatch_n_tok - (int32_t) nt;
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    gdn_capture_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
    for (int j = 0; j < nt; ++j) {
        const int col = base + j;
        if (col < 0 || col >= (int) col_pos.size() || col_pos[col] < 0) { continue; }
        const int pos = col_pos[col];
        if (pos >= gdn_max_pos) { l.over_cap = true; continue; }
        std::vector<float> row(width);
        for (int d = 0; d < width; ++d) {
            const size_t offset = headed ? (d%t->ne[0])*t->nb[0] + (d/t->ne[0])*t->nb[1] + j*t->nb[2] : d*t->nb[0] + j*t->nb[1];
            const auto * p = buf.data() + offset;
            if (t->type == GGML_TYPE_F32) { memcpy(&row[d], p, sizeof(float)); }
            else if (t->type == GGML_TYPE_F16) { ggml_fp16_t x; memcpy(&x, p, sizeof(x)); row[d] = ggml_fp16_to_fp32(x); }
            else { ggml_bf16_t x; memcpy(&x, p, sizeof(x)); row[d] = ggml_bf16_to_fp32(x); }
        }
        if (kind == TRACE_KIND_GDN_Q) { l.q[col] = std::move(row); }
        else if (kind == TRACE_KIND_GDN_READ) { l.read[col] = std::move(row); }
        else if (kind == TRACE_KIND_GDN_Z) { l.z[col] = std::move(row); }
        else if (kind == TRACE_KIND_GDN_Y) { l.y[col] = std::move(row); }
        else if (kind == TRACE_KIND_GDN_OUT) { l.output[col] = std::move(row); }
        else {
            l.present.resize(std::max(l.present.size(), (size_t) pos + 1), 0);
            if (kind == TRACE_KIND_GDN_K || kind == TRACE_KIND_GDN_V) {
                auto & dst = kind == TRACE_KIND_GDN_K ? l.k : l.v;
                dst.resize(std::max(dst.size(), (pos + 1)*row.size()));
                ggml_fp32_to_fp16_row(row.data(), dst.data() + pos*width, width);
                l.present[pos] |= kind == TRACE_KIND_GDN_K ? 1 : 2;
            } else {
                auto & dst = kind == TRACE_KIND_GDN_BETA ? l.beta : l.a;
                dst.resize(std::max(dst.size(), (pos + 1)*row.size()));
                for (int d = 0; d < width; ++d) { dst[pos*width + d] = kind == TRACE_KIND_GDN_GATE ? std::exp(row[d]) : row[d]; }
                l.present[pos] |= kind == TRACE_KIND_GDN_BETA ? 4 : 8;
            }
        }
    }
}

static double trace_gdn_round(double x) {
    if (!std::isfinite(x) || x == 0) { return 0; }
    char text[32];
    snprintf(text, sizeof(text), "%.6g", x);
    return strtod(text, nullptr);
}

static double trace_gdn_error(const float * a, const float * b, int n) {
    double diff = 0, norm = 0;
    for (int j = 0; j < n; ++j) { diff += ((double) a[j] - b[j])*(a[j] - b[j]); norm += (double) b[j]*b[j]; }
    return std::sqrt(diff/std::max(norm, 1e-30));
}

json server_moe_trace::gdn_json(int32_t col, int64_t n_pos, int32_t top_k) {
    static const bool selfcheck = [] { const char * e = getenv("LLAMA_TRACE_GDN_SELFCHECK"); return e && strcmp(e, "1") == 0; }();
    struct result { json share, heads, gates, fade; std::string note; double a = 0, b = 0, c = 0, y = 0, sum = 0; };
    std::vector<result> results(gdn.size());
    auto compute = [&](size_t index) {
        const auto & l = gdn[index];
        auto & out = results[index];
        if (n_pos > gdn_max_pos || l.over_cap) {
            out.note = "gdn skipped: n_pos " + std::to_string(n_pos) + " exceeds max_pos " + std::to_string(gdn_max_pos);
            return;
        }
        if (n_pos <= 0 || n_pos > (int64_t) l.present.size() ||
                std::any_of(l.present.begin(), l.present.begin() + n_pos, [](unsigned char mask) { return mask != 15; }) ||
                !l.q.count(col) || !l.read.count(col) || !l.z.count(col) || !l.y.count(col) || !l.output.count(col)) {
            out.note = "gdn skipped: missing captured positions or current column";
            return;
        }
        const int n = (int) n_pos, width = l.s*l.h;
        const auto & read = l.read.at(col);
        const auto & output = l.output.at(col);
        const auto weights = server_trace_gdn_sweep(l.s, l.h, l.kh, n, l.k.data(), l.beta.data(), l.a.data(), l.q.at(col).data());
        const auto factors = server_trace_gdn_norm(l.s, l.h, read.data(), l.norm.data(), l.norm.size() == (size_t) l.s, l.z.at(col).data(), l.eps);
        std::vector<double> shares, heads;
        server_trace_gdn_shares(l.s, l.h, n, l.out_dim, l.v.data(), weights.data(), read.data(), factors.data(), l.out_weight.data(), output.data(), shares, heads);
        out.sum = std::abs(std::accumulate(shares.begin(), shares.end(), 0.0) - 1.0);
        out.share = json::array(); out.heads = json::array(); out.gates = json::array(); out.fade = json::array();
        for (double share : shares) { out.share.push_back(trace_gdn_round(share)); }
        std::vector<double> fades(n, 0);
        for (int head = 0; head < l.h; ++head) {
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return std::abs(heads[a*l.h + head]) > std::abs(heads[b*l.h + head]); });
            json top = json::array();
            for (int j = 0; j < std::min(n, top_k); ++j) { top.push_back(json::array({order[j], trace_gdn_round(heads[order[j]*l.h + head])})); }
            out.heads.push_back(std::move(top));
            out.gates.push_back(json::array({trace_gdn_round(l.beta[(n - 1)*l.h + head]), trace_gdn_round(l.a[(n - 1)*l.h + head])}));
            double fade = 1;
            for (int i = n - 1; i >= 0; --i) { fades[i] += fade/l.h; fade *= l.a[i*l.h + head]; }
        }
        for (double fade : fades) { out.fade.push_back(trace_gdn_round(fade)); }
        if (selfcheck) {
            const auto forward = server_trace_gdn_forward(l.s, l.h, l.kh, n, l.k.data(), l.v.data(), l.beta.data(), l.a.data(), l.q.at(col).data());
            std::vector<float> summed(width, 0), y(width), projected(l.out_dim, 0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < width; ++j) { summed[j] += weights[i*l.h + j/l.s]*ggml_fp16_to_fp32(l.v[i*width + j]); }
            }
            for (int head = 0; head < l.h; ++head) {
                out.a = std::max(out.a, trace_gdn_error(summed.data() + head*l.s, forward.data() + head*l.s, l.s));
                out.b = std::max(out.b, trace_gdn_error(forward.data() + head*l.s, read.data() + head*l.s, l.s));
            }
            for (int j = 0; j < width; ++j) { y[j] = read[j]*factors[j]; }
            for (int r = 0; r < l.out_dim; ++r) {
                for (int j = 0; j < width; ++j) { projected[r] += ggml_fp16_to_fp32(l.out_weight[r*width + j])*y[j]; }
            }
            out.c = trace_gdn_error(projected.data(), output.data(), l.out_dim);
            out.y = trace_gdn_error(y.data(), l.y.at(col).data(), width);
        }
    };
    static const bool timing = [] { const char * e = getenv("LLAMA_TRACE_GDN_TIMING"); return e && strcmp(e, "1") == 0; }();
    const auto t_start = std::chrono::steady_clock::now();
    const size_t count = std::min<size_t>(16, gdn.size());
    std::vector<std::thread> workers;
    for (size_t worker = 0; worker < count; ++worker) {
        workers.emplace_back([&, worker] { for (size_t i = worker; i < gdn.size(); i += count) { compute(i); } });
    }
    for (auto & worker : workers) { worker.join(); }
    if (timing) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        LOG_INF("gdn timing: n_pos=%lld compute %.1f ms over %zu blocks, capture %.1f ms in this batch\n", (long long) n_pos, ms, gdn.size(), gdn_capture_us/1000.0);
    }
    json out = {{"blocks", json::array()}, {"share", json::array()}, {"heads", json::array()}, {"gates", json::array()}, {"fade", json::array()}};
    double a = 0, b = 0, c = 0, y = 0, sum = 0;
    for (size_t i = 0; i < gdn.size(); ++i) {
        auto & r = results[i];
        out["blocks"].push_back(gdn[i].il);
        out["share"].push_back(std::move(r.share)); out["heads"].push_back(std::move(r.heads));
        out["gates"].push_back(std::move(r.gates)); out["fade"].push_back(std::move(r.fade));
        if (!r.note.empty()) { out["note"] = r.note; }
        a = std::max(a, r.a); b = std::max(b, r.b); c = std::max(c, r.c); y = std::max(y, r.y); sum = std::max(sum, r.sum);
    }
    if (selfcheck) {
        LOG_INF("gdn selfcheck pos=%lld max_a=%.6g max_b=%.6g max_c=%.6g max_sum=%.6g max_y=%.6g\n", (long long) n_pos - 1, a, b, c, sum, y);
    }
    return out;
}
