#include "idx-relu-sum.cuh"

// Preserve the head addition order of the RELU/CONT/ADD graph.
// With tails/starts it also applies the compact QSA visibility exactly as CPY(I32->F32), REPEAT, SUB, STEP, LOG
// and ADD compute it, so the result is bit-identical to the unfused graph.
static __global__ void idx_relu_sum_f32(const float * __restrict__ src, float * __restrict__ dst,
                                        const int n_blocks, const int heads,
                                        const int32_t * __restrict__ tails, const int32_t * __restrict__ starts) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const int64_t tt = blockIdx.y;                         // token within stream, streams stacked
    const float * p  = src + tt * (int64_t) heads * n_blocks + b;
    float acc = fmaxf(p[0], 0.0f);
    for (int h = 1; h < heads; ++h) {
        acc = acc + fmaxf(p[(int64_t) h * n_blocks], 0.0f);
    }
    if (tails) {
        const float d    = (float) tails[tt] - (float) starts[b];
        const float step = d > 0.0f;
        acc = acc + logf(step);
    }
    dst[tt * (int64_t) n_blocks + b] = acc;
}

void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args) {
    const ggml_tensor * s = args.score;
    const int n_blocks = (int) s->ne[0];
    const int64_t rows = s->ne[2] * s->ne[3];
    constexpr int threads = 256;
    const dim3 grid((n_blocks + threads - 1) / threads, (unsigned) rows, 1);
    const int32_t * tails  = args.vis_tails  ? (const int32_t *) args.vis_tails->data  : nullptr;
    const int32_t * starts = args.vis_starts ? (const int32_t *) args.vis_starts->data : nullptr;
    idx_relu_sum_f32<<<grid, threads, 0, ctx.stream()>>>((const float *) s->data, (float *) args.dst->data, n_blocks, args.heads,
        tails, starts);
    CUDA_CHECK(cudaGetLastError());
}

// qwen4exp QSA indexer scorer at decode (1-8 queries, n_stream 1): MUL_MAT(pooled keys F32 [128, M], q F32 [128, 4*T])
// -> RESHAPE [M, 4, T] -> RELU -> 4-head sum (-> compact visibility) in one launch, writing dst F32 [M, T].
// The unfused MUL_MAT takes mul_mat_vec_f<float, float, 4*T, 64> for 4*T <= 8 columns (K = 128: one K step per lane, two
// waves per row). Every dot product here uses its reduction tree: lane l accumulates columns 2l, 2l+1 with ggml_cuda_mad,
// each wave butterflies with warp_reduce_sum, and the block reduce reads [wave0, wave1, 0 x 30] into one wave and
// butterflies again. The RELU / head-sum / visibility epilogue is idx_relu_sum_f32's. So at 1-2 queries the scores are
// bit-identical to the unfused graph; at 3-8 queries the unfused GEMM is rocBLAS (other summation order), and the scores
// equal what the same queries get one or two at a time instead.
// A block is 4 groups of 64 threads; each group keeps its q columns in registers and walks IDXD_RPG rows.
static constexpr int IDXD_K = 128, IDXD_GROUPS = 4, IDXD_RPG = 4;

template <int NQ>
static __global__ void __launch_bounds__(64 * IDXD_GROUPS, 1)
idx_score_decode_f32(const float * __restrict__ keys, const float * __restrict__ q, float * __restrict__ dst,
                     const int M, const int64_t stride_key, const int64_t stride_q,
                     const int32_t * __restrict__ tails, const int32_t * __restrict__ starts) {
    constexpr int NC = 4 * NQ;
    constexpr int warp_size = 32;
    __shared__ float red[IDXD_GROUPS][NC][2];
    __shared__ float res[IDXD_GROUPS][NC];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, g = threadIdx.y;
    const int row0 = (blockIdx.x * IDXD_GROUPS + g) * IDXD_RPG;

    float2 y[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        y[j] = ((const float2 *) (q + j * stride_q))[tid];
    }
    float2 x[IDXD_RPG];
#pragma unroll
    for (int r = 0; r < IDXD_RPG; ++r) {
        x[r] = row0 + r < M ? ((const float2 *) (keys + (int64_t) (row0 + r) * stride_key))[tid] : make_float2(0.0f, 0.0f);
    }

#pragma unroll 1
    for (int r = 0; r < IDXD_RPG; ++r) {
        const int row = row0 + r;
        float sumf[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            sumf[j] = 0.0f;
            ggml_cuda_mad(sumf[j], x[r], y[j]);
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                red[g][j][wave] = sumf[j];
            }
        }
        __syncthreads();
        // mul_mat_vec_f's block reduce: lanes 0, 1 hold the wave sums, the other lanes the zeroed buffer
#pragma unroll
        for (int j = wave; j < NC; j += 2) {
            float v = lane < 2 ? red[g][j][lane] : 0.0f;
            v = warp_reduce_sum<warp_size>(v);
            if (lane == 0) {
                res[g][j] = v;
            }
        }
        __syncthreads();
        if (tid < NQ && row < M) {
            const float * s = res[g] + 4 * tid;
            float acc = fmaxf(s[0], 0);
#pragma unroll
            for (int h = 1; h < 4; ++h) {
                acc = acc + fmaxf(s[h], 0);
            }
            if (tails) {
                const float d    = (float) tails[tid] - (float) starts[row];
                const float step = d > 0.0f;
                acc = acc + logf(step);
            }
            dst[(int64_t) tid * M + row] = acc;
        }
    }
}

int ggml_cuda_idx_score_decode_mode() {
    // 0: off (unfused chain), 1: 1-8 queries, 2: only where the unfused GEMM is the matvec (1-2 queries, bit-identical)
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_FUSE_IDX_DECODE");
        return env ? atoi(env) : 1;
    }();
    return mode;
}

bool ggml_cuda_idx_score_decode_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * mm) {
    const int mode = ggml_cuda_idx_score_decode_mode();
    if (mode <= 0) return false;
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(mm)) return false;
    if (src0->ne[0] != IDXD_K || src1->ne[0] != IDXD_K || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    const int64_t cols = src1->ne[1];
    if (cols % 4 != 0 || cols < 4 || cols > (mode == 2 ? 8 : 32)) return false;
    if (src0->ne[1] > INT32_MAX / 4 || mm->ne[0] != src0->ne[1] || mm->ne[1] != cols) return false;
    return ((uintptr_t) src0->data % 8) == 0 && ((uintptr_t) src1->data % 8) == 0;
}

void ggml_cuda_idx_score_decode(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * tails, const int32_t * starts) {
    const int M  = (int) src0->ne[1];
    const int NQ = (int) (src1->ne[1] / 4);
    const float * keys = (const float *) src0->data, * q = (const float *) src1->data;
    float * d = (float *) dst->data;
    const int64_t sk = src0->nb[1] / sizeof(float), sq = src1->nb[1] / sizeof(float);
    constexpr int rows_per_block = IDXD_GROUPS * IDXD_RPG;
    const dim3 grid((M + rows_per_block - 1) / rows_per_block, 1, 1), block(64, IDXD_GROUPS, 1);
    cudaStream_t stream = ctx.stream();
    switch (NQ) {
        case 1: idx_score_decode_f32<1><<<grid, block, 0, stream>>>(keys, q, d, M, sk, sq, tails, starts); break;
        case 2: idx_score_decode_f32<2><<<grid, block, 0, stream>>>(keys, q, d, M, sk, sq, tails, starts); break;
        case 3: idx_score_decode_f32<3><<<grid, block, 0, stream>>>(keys, q, d, M, sk, sq, tails, starts); break;
        case 4: idx_score_decode_f32<4><<<grid, block, 0, stream>>>(keys, q, d, M, sk, sq, tails, starts); break;
        case 5: idx_score_decode_f32<5><<<grid, block, 0, stream>>>(keys, q, d, M, sk, sq, tails, starts); break;
        case 6: idx_score_decode_f32<6><<<grid, block, 0, stream>>>(keys, q, d, M, sk, sq, tails, starts); break;
        case 7: idx_score_decode_f32<7><<<grid, block, 0, stream>>>(keys, q, d, M, sk, sq, tails, starts); break;
        case 8: idx_score_decode_f32<8><<<grid, block, 0, stream>>>(keys, q, d, M, sk, sq, tails, starts); break;
        default: GGML_ABORT("idx_score_decode: %d queries", NQ);
    }
    CUDA_CHECK(cudaGetLastError());
}
