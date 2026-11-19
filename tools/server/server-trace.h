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
// A generated token's entry is the column its logits came from, so the trace also works under speculative
// decoding, where one verify batch yields several accepted tokens.

#include "server-common.h"

#include "llama.h"
#include "ggml.h"

#include <vector>

struct server_moe_trace {
    int32_t n_layer   = 0;
    bool    installed = false;
    bool    want_block_update = false; // capture the residuals too (more data per layer, slower while on)

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

    std::vector<layer_capture> layers;
    std::vector<resid_capture> resid;     // [0] = hc_init, [il + 1] = l_last-<il>
    int32_t ubatch_base   = 0;            // batch column of the current ubatch's first token (from hc_init)
    int32_t ubatch_n_tok  = 0;

    void init(int32_t n_layer);

    // install/remove the eval callback on the target context
    void install(llama_context * ctx);
    void remove (llama_context * ctx);

    // drop the captures of the previous batch; call before every decode of the target context
    void begin_batch();

    // {"token_id": tok, "moe": [per layer {"experts", "weights", "shared"} or null], "block_update": [per layer f or null]}
    // col is the token's column in the decoded batch; experts selects the moe part, block_update the residual part
    json token_json(int32_t col, llama_token tok, bool experts, bool block_update) const;

    static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

private:
    void capture(struct ggml_tensor * t, int kind, int32_t il);
    void capture_resid(struct ggml_tensor * t, int32_t idx);
};
