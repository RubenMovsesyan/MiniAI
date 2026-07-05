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

Matrix softmax(const Matrix& x) {
    RLOG(LL_WARN, "softmax not yet implemented");
    return Matrix(x.rows(), x.cols());
}

void softmax(const Matrix& x, Matrix& out) {
    RLOG(LL_WARN, "softmax not yet implemented");
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
