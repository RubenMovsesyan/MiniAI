#include <rlog.h>
#include <nn/losses.cuh>
#include <nn/activations.cuh>
#include <agg/agg.cuh>

Matrix mse(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "mse not yet implemented");
    return Matrix(1, 1);
}

// ─── Cross-entropy ──────────────────────────────────────────────────────────────
// CE = -(1/N) Σ_i Σ_j targets[i,j] * log(a2[i,j] + eps), a2 = softmax(logits), N = rows.
// One block per row reduces -Σ_j t*log(a2+eps) into rowCE[r] (shared-mem tree reduction,
// same shape as softmaxKernel pass 1), then col_sum → scalar, then scale by 1/N.

__global__ static void crossEntropyRowKernel(const f32* a2, const f32* targets,
                                             f32* rowCE, i32 rows, i32 cols) {
    extern __shared__ f32 s[];
    i32 row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;

    f32 acc = 0.0f;
    for (i32 c = tid; c < cols; c += blockDim.x) {
        i32 idx = row * cols + c;
        acc += targets[idx] * logf(a2[idx] + 1e-9f);
    }
    s[tid] = acc; __syncthreads();
    for (i32 st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) rowCE[row] = -s[0];
}

Matrix cross_entropy(const Matrix& logits, const Matrix& targets) {
    i32 N = logits.rows();
    Matrix a2 = softmax(logits);
    Matrix rowCE(N, 1);
    crossEntropyRowKernel<<<N, 256, 256 * sizeof(f32), g_compute_stream>>>(
        a2.data, targets.data, rowCE.data, N, logits.cols());
    Matrix total = col_sum(rowCE);          // (N,1) → (1,1)
    Matrix loss(1, 1);
    loss = total * (1.0f / (f32)N);
    return loss;
}

Matrix l1_loss(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "l1_loss not yet implemented");
    return Matrix(1, 1);
}

Matrix l2_loss(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "l2_loss not yet implemented");
    return Matrix(1, 1);
}
