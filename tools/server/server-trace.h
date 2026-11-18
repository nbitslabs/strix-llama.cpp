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
// A generated token's entry is the column its logits came from, so the trace also works under speculative
// decoding, where one verify batch yields several accepted tokens.

#include "server-common.h"

#include "llama.h"
#include "ggml.h"

#include <vector>

struct server_moe_trace {
    int32_t n_layer   = 0;
    bool    installed = false;

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

    std::vector<layer_capture> layers;

    void init(int32_t n_layer);

    // install/remove the eval callback on the target context
    void install(llama_context * ctx);
    void remove (llama_context * ctx);

    // drop the captures of the previous batch; call before every decode of the target context
    void begin_batch();

    // {"token_id": tok, "moe": [per layer {"experts": [...], "weights": [...], "shared": f} or null]}
    // col is the token's column in the decoded batch
    json token_json(int32_t col, llama_token tok) const;

    static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

private:
    void capture(struct ggml_tensor * t, int kind, int32_t il);
};
