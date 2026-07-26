#include <rlog.h>
#include <nn/activations.cuh>

// ─── Relu ─────────────────────────────────────────────────────────────────────

__global__ static void reluKernel(const f32* x, f32* out, i32 rows, i32 cols) {
    i32 r = blockIdx.y * 16 + threadIdx.y;
    i32 c = blockIdx.x * 16 + threadIdx.x;
    if (r < rows && c < cols)
        out[r * cols + c] = fmaxf(0.0f, x[r * cols + c]);
}

Matrix relu(const Matrix& x) {
    Matrix out(x.rows(), x.cols());
    dim3 block(16, 16);
    dim3 grid((x.cols() + 15) / 16, (x.rows() + 15) / 16);
    reluKernel<<<grid, block, 0, g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
    return out;
}

void relu(const Matrix& x, Matrix& out) {
    dim3 block(16, 16);
    dim3 grid((x.cols() + 15) / 16, (x.rows() + 15) / 16);
    reluKernel<<<grid, block, 0, g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
}

// ─── Sigmoid ───────────────────────────────────────────────────────────────────

Matrix sigmoid(const Matrix& x) {
    RLOG(LL_WARN, "sigmoid not yet implemented");
    return Matrix(x.rows(), x.cols());
}

void sigmoid(const Matrix& x, Matrix& out) {
    RLOG(LL_WARN, "sigmoid not yet implemented");
}

// ─── Bipolar Sigmoid ──────────────────────────────────────────────────────────

Matrix bipolar_sigmoid(const Matrix& x) {
    RLOG(LL_WARN, "bipolar_sigmoid not yet implemented");
    return Matrix(x.rows(), x.cols());
}

void bipolar_sigmoid(const Matrix& x, Matrix& out) {
    RLOG(LL_WARN, "bipolar_sigmoid not yet implemented");
}

// ─── Tanh ──────────────────────────────────────────────────────────────────────

Matrix tanh(const Matrix& x) {
    RLOG(LL_WARN, "tanh not yet implemented");
    return Matrix(x.rows(), x.cols());
}

void tanh(const Matrix& x, Matrix& out) {
    RLOG(LL_WARN, "tanh not yet implemented");
}

// ─── Leaky ReLU ────────────────────────────────────────────────────────────────

Matrix leaky_relu(const Matrix& x, f32 alpha) {
    RLOG(LL_WARN, "leaky_relu not yet implemented");
    return Matrix(x.rows(), x.cols());
}

void leaky_relu(const Matrix& x, f32 alpha, Matrix& out) {
    RLOG(LL_WARN, "leaky_relu not yet implemented");
}

// ─── Softmax ───────────────────────────────────────────────────────────────────
// Row-wise softmax — a reduction, not element-wise, so it needs a dedicated kernel.
// One block per row, 256 threads, shared-memory tree reduction. Four passes:
// (1) row max, (2+3) exp(x - max) written to out + row sum, (4) divide by sum.
// Max-subtraction keeps expf from overflowing f32 to inf on large logits.

__global__ static void softmaxKernel(const f32* x, f32* out, i32 rows, i32 cols) {
    extern __shared__ f32 s[];
    i32 row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;

    // pass 1: row max (stability)
    f32 m = -INFINITY;
    for (i32 c = tid; c < cols; c += blockDim.x) m = fmaxf(m, x[row * cols + c]);
    s[tid] = m; __syncthreads();
    for (i32 st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] = fmaxf(s[tid], s[tid + st]);
        __syncthreads();
    }
    f32 rowmax = s[0]; __syncthreads();

    // pass 2+3: write exp(x - max), accumulate row sum
    f32 sum = 0.0f;
    for (i32 c = tid; c < cols; c += blockDim.x) {
        f32 e = expf(x[row * cols + c] - rowmax);
        out[row * cols + c] = e;
        sum += e;
    }
    s[tid] = sum; __syncthreads();
    for (i32 st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    f32 rowsum = s[0]; __syncthreads();

    // pass 4: normalize
    for (i32 c = tid; c < cols; c += blockDim.x) out[row * cols + c] /= rowsum;
}

// Launch helper — lets the lazy materialize path (in activations.cuh) invoke the
// kernel without seeing its definition (kernel stays in this TU).
void softmaxLaunch(const f32* x, f32* out, i32 rows, i32 cols) {
    softmaxKernel<<<rows, 256, 256 * sizeof(f32), g_compute_stream>>>(x, out, rows, cols);
}

Matrix softmax(const Matrix& x) {
    Matrix out(x.rows(), x.cols());
    softmaxLaunch(x.data, out.data, x.rows(), x.cols());
    return out;
}

void softmax(const Matrix& x, Matrix& out) {
    softmaxLaunch(x.data, out.data, x.rows(), x.cols());
}

// ─── Step ──────────────────────────────────────────────────────────────────────

Matrix step(const Matrix& x, f32 threshold) {
    RLOG(LL_WARN, "step not yet implemented");
    return Matrix(x.rows(), x.cols());
}

void step(const Matrix& x, f32 threshold, Matrix& out) {
    RLOG(LL_WARN, "step not yet implemented");
}

// ─── Threshold ─────────────────────────────────────────────────────────────────

Matrix threshold(const Matrix& x, f32 thresh) {
    RLOG(LL_WARN, "threshold not yet implemented");
    return Matrix(x.rows(), x.cols());
}

void threshold(const Matrix& x, f32 thresh, Matrix& out) {
    RLOG(LL_WARN, "threshold not yet implemented");
}
