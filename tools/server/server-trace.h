#pragma once

// Opt-in per-token MoE router trace for the completion endpoints (request field "trace": {"experts": true}).
//
// Local feature of this fork's server, not for upstream (tools/server/README-dev.md excludes API access to
// intermediate activations). The capture goes through the scheduler eval callback, which is installed on the
// target context only while a traced request runs, so untraced requests keep the normal graph path.
//
// Captured per layer, for every column of each decoded batch:
//   ffn_moe_topk-<il>                the expert ids the router selected (engine order, not sorted)
//   ffn_moe_weights_norm-<il>        the normalized gate weights that multiply the expert outputs
//   shared_expert_gate_sigmoid-<il>  the shared expert's gate (one value per token), when the model has one
// With "block_update": true the wide residual after every layer (l_last-<il>, hc_init before layer 0) is captured as
// well, and each token reports ||l_last[il] - l_last[il-1]|| / ||l_last[il-1]|| over all streams at its column.
// With "block_push": true the same residuals go through a logit lens on the CPU (the model's own final hc mixer, then
// the output rows of the sampled token and its top candidates), and each token reports the per-block logit change.
// With "checkpoints": {"after_blocks": [...], "top_k": 8}, selected residuals use the same lens over the full vocabulary;
// the row after the last block uses the engine's own head input (result_norm), so it reproduces the engine's distribution.
// A same-type CPU copy of the output head and a reusable matmul graph give top probabilities, sampled-token rank and entropy.
// A generated token's entry is the column its logits came from, so the trace also works under speculative
// decoding, where one verify batch yields several accepted tokens.

#include "server-common.h"

#include "llama.h"
#include "ggml.h"

#include <memory>
#include <unordered_map>
#include <vector>

struct server_trace_head; // CPU copy of the output head for checkpoints, defined in server-trace.cpp

struct server_moe_trace {
    int32_t n_layer   = 0;
    int64_t n_embd    = 0;
    int64_t n_stream  = 1;     // hyper-connection streams of the wide residual (n_embd_out / n_embd)
    bool    installed = false;
    bool    want_block_update = false; // capture the residuals too (more data per layer, slower while on)
    bool    want_attention    = false;
    bool    want_head_in      = false; // capture the head input (result_norm) for checkpoints' last-block row
    std::vector<char> resid_wanted;    // per resid index; empty = every residual (block_update / block_push)

    // logit lens for block_push: the final hc mixer weights (dequantized once) and the output rows asked for so far
    struct lens_weights {
        bool    ready  = false;
        int64_t n_embd = 0;
        int64_t hc     = 0;
        int64_t hc_lr  = 0;
        float   eps    = 1e-6f;
        float   out_scale = 1.0f;
        std::vector<float> norm; // [hc][n_embd]
        std::vector<float> down; // [hc_lr][hc*n_embd]
        std::vector<float> up;   // [hc*n_embd][hc_lr]
        struct ggml_tensor * output = nullptr;
        std::unordered_map<llama_token, std::vector<float>> rows;
        std::shared_ptr<server_trace_head> head;
    };
    lens_weights lens;

    struct layer_capture {
        int32_t n_used_experts = 0;
        int32_t n_used_weights = 0;
        int32_t n_tokens_experts = 0;
        int32_t n_tokens_weights = 0;
        int32_t n_tokens_shared  = 0;
        std::vector<int32_t> experts; // [n_tokens][n_used_experts]
        std::vector<float>   weights; // [n_tokens][n_used_weights]
        std::vector<float>   shared;  // [n_tokens]
    };

    struct resid_capture {
        int64_t rows = 0;                 // n_embd * n_streams
        std::vector<int32_t> cols;        // batch column index of each captured column
        std::vector<float>   data;        // [cols.size()][rows]
    };

    struct attention_capture {
        int64_t n_head = 0;
        int64_t n_kv = 0;
        std::vector<int32_t> cols;
        std::vector<std::vector<float>> data; // [column][head][kv cell]; KV extent can change between ubatches
    };
    std::vector<attention_capture> attention;

    std::vector<layer_capture> layers;
    std::vector<resid_capture> resid;     // [0] = hc_init, [il + 1] = l_last-<il>
    resid_capture head_in;                // [n_embd] per column: the final mixer output the engine's head consumed (result_norm)
    int32_t ubatch_base   = 0;            // batch column of the current ubatch's first token (from hc_init)
    int32_t ubatch_n_tok  = 0;

    void init(int32_t n_layer, int64_t n_embd, int64_t n_stream);
    // restrict the residual capture to hc_init + the listed blocks (nullptr = every residual); hc_init stays because it
    // carries the ubatch column bookkeeping
    void resid_filter(const std::vector<int32_t> * blocks);

    // install/remove the eval callback on the target context
    void install(llama_context * ctx);
    void remove (llama_context * ctx);

    // drop the captures of the previous batch; call before every decode of the target context
    void begin_batch();

    // {"token_id": tok, "moe": [per layer {"experts", "weights", "shared"} or null], "block_update": [per layer f or null]}
    // col is the token's column in the decoded batch; experts selects the moe part, block_update the residual part
    json token_json(int32_t col, llama_token tok, bool experts, bool block_update) const;

    // load the final mixer weights of the model (once); false with an error message when the model has none
    bool lens_prepare(const llama_model * model, std::string & err, bool checkpoints = false);

    json attention_json(int32_t col, int64_t n_pos) const;

    json checkpoints_json(int32_t col, llama_token tok, const std::vector<int32_t> & blocks, int32_t top_k);

    // {"token_ids": [...], "base": [...], "push": [[n_layer] per token], "logit": [...]} for the residuals at col;
    // tokens[0] is the sampled token, logits[t] the engine's final logit of tokens[t] (NaN when not available)
    json block_push_json(int32_t col, const std::vector<llama_token> & tokens, const std::vector<float> & logits);

    static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

private:
    bool lens_prepare_head(std::string & err);
    void capture(struct ggml_tensor * t, int kind, int32_t il);
    void capture_resid(struct ggml_tensor * t, int32_t idx);
    void capture_attention(struct ggml_tensor * t, int32_t il);
    void capture_head_in(struct ggml_tensor * t);
    const std::vector<float> & lens_row(llama_token tok);
    // the lens hidden state (final hc mixer output, [n_embd]) of one wide residual [hc][n_embd]
    void lens_hidden(const float * x, std::vector<float> & out) const;
};
