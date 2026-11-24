#include "server-trace.h"

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

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
    if (strcmp(name, "h_nextn") == 0) {
        il = -2; // the last layer's l_last, renamed when it is handed to the MTP head; mapped to n_layer below
        return TRACE_KIND_RESID;
    }
    return TRACE_KIND_NONE;
}

void server_moe_trace::init(int32_t n_layer_, int64_t n_embd_, int64_t n_stream_) {
    n_layer  = n_layer_;
    n_embd   = n_embd_;
    n_stream = n_stream_ > 0 ? n_stream_ : 1;
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
    lens.rows.clear(); // the dequantized output rows are per request; the mixer weights stay
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
    int kind = trace_classify(ggml_get_name(t), il);
    if (kind == TRACE_KIND_RESID && il == -2) {
        il = self->n_layer;
    }
    const bool wanted = kind == TRACE_KIND_RESID ? (self->want_block_update && il >= 0 && il <= self->n_layer)
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
    std::vector<float> h;
    for (int32_t r = 0; r <= n_layer; ++r) {
        const auto & R = resid[r];
        const float * x = R.rows == lens.hc*lens.n_embd ? trace_resid_col(R, col) : nullptr;
        if (!x) {
            continue;
        }
        lens_hidden(x, h);
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
    for (int32_t il : blocks) {
        const auto & R = resid.at(il + 1);
        const float * x = R.rows == lens.hc*lens.n_embd ? trace_resid_col(R, col) : nullptr;
        if (!x) {
            for (const char * key : {"top", "final_prob", "final_rank", "entropy"}) {
                out[key].push_back(nullptr);
            }
            continue;
        }
        lens_hidden(x, h);
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
