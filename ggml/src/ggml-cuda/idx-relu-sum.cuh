#include "common.cuh"

struct ggml_cuda_idx_relu_sum_args {
    const ggml_tensor * score = nullptr;
    ggml_tensor *       dst   = nullptr;   // [n_blocks, n_tps, n_stream]
    int                 heads = 0;
    // optional fused compact visibility (single sequence): dst += log(step(float(tails[t]) - float(starts[b])))
    const ggml_tensor * vis_tails  = nullptr;   // I32 [n_tps] contiguous view of the QSA limits
    const ggml_tensor * vis_starts = nullptr;   // I32 [n_blocks] contiguous view of the QSA limits
    // nodes of the visibility chain that the matcher checked itself (views/casts of the limits input); a caller that
    // widens the fusion must leave them out of ggml_can_fuse_subgraph_ext
    const ggml_tensor * vis_leafs[16] = {};
};
void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args);

// qwen4exp QSA indexer scorer at decode: MUL_MAT(keys F32 [128, M], q F32 [128, 4*T]) + RELU + 4-head sum (+ visibility),
// T = 1..8, n_stream 1. GGML_CUDA_FUSE_IDX_DECODE: 0 off, 1 T = 1..8, 2 (default) only T = 1..2 (bit-identical to the matvec)
int  ggml_cuda_idx_score_decode_mode();
bool ggml_cuda_idx_score_decode_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * mm);
void ggml_cuda_idx_score_decode(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * tails, const int32_t * starts);
