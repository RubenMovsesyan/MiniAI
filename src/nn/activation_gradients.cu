#include <rlog.h>
#include <nn/activation_gradients.cuh>

// ─── Grad ReLU ────────────────────────────────────────────────────────────────

__global__ static void gradReluKernel(const f32* x, const f32* dy, f32* out, i32 rows, i32 cols) {
    i32 r = blockIdx.y * 16 + threadIdx.y;
    i32 c = blockIdx.x * 16 + threadIdx.x;
    if (r < rows && c < cols)
        out[r * cols + c] = x[r * cols + c] > 0.0f ? dy[r * cols + c] : 0.0f;
}

Matrix grad_relu(const Matrix& x, const Matrix& dy) {
    Matrix out(x.rows(), x.cols());
    dim3 block(16, 16);
    dim3 grid((x.cols() + 15) / 16, (x.rows() + 15) / 16);
    gradReluKernel<<<grid, block, 0, g_compute_stream>>>(x.data, dy.data, out.data, x.rows(), x.cols());
    return out;
}

void grad_relu(const Matrix& x, const Matrix& dy, Matrix& out) {
    dim3 block(16, 16);
    dim3 grid((x.cols() + 15) / 16, (x.rows() + 15) / 16);
    gradReluKernel<<<grid, block, 0, g_compute_stream>>>(x.data, dy.data, out.data, x.rows(), x.cols());
}

// ─── Grad Sigmoid ──────────────────────────────────────────────────────────────

Matrix grad_sigmoid(const Matrix& y, const Matrix& dy) {
    RLOG(LL_WARN, "grad_sigmoid not yet implemented");
    return Matrix(y.rows(), y.cols());
}

void grad_sigmoid(const Matrix& y, const Matrix& dy, Matrix& out) {
    RLOG(LL_WARN, "grad_sigmoid not yet implemented");
}

// ─── Grad Bipolar Sigmoid ─────────────────────────────────────────────────────

Matrix grad_bipolar_sigmoid(const Matrix& y, const Matrix& dy) {
    RLOG(LL_WARN, "grad_bipolar_sigmoid not yet implemented");
    return Matrix(y.rows(), y.cols());
}

void grad_bipolar_sigmoid(const Matrix& y, const Matrix& dy, Matrix& out) {
    RLOG(LL_WARN, "grad_bipolar_sigmoid not yet implemented");
}

// ─── Grad Tanh ────────────────────────────────────────────────────────────────

Matrix grad_tanh(const Matrix& y, const Matrix& dy) {
    RLOG(LL_WARN, "grad_tanh not yet implemented");
    return Matrix(y.rows(), y.cols());
}

void grad_tanh(const Matrix& y, const Matrix& dy, Matrix& out) {
    RLOG(LL_WARN, "grad_tanh not yet implemented");
}

// ─── Grad Leaky ReLU ──────────────────────────────────────────────────────────

Matrix grad_leaky_relu(const Matrix& x, const Matrix& dy, f32 alpha) {
    RLOG(LL_WARN, "grad_leaky_relu not yet implemented");
    return Matrix(x.rows(), x.cols());
}

void grad_leaky_relu(const Matrix& x, const Matrix& dy, f32 alpha, Matrix& out) {
    RLOG(LL_WARN, "grad_leaky_relu not yet implemented");
}

// ─── Grad Softmax ─────────────────────────────────────────────────────────────

Matrix grad_softmax(const Matrix& y, const Matrix& dy) {
    RLOG(LL_WARN, "grad_softmax not yet implemented");
    return Matrix(y.rows(), y.cols());
}

void grad_softmax(const Matrix& y, const Matrix& dy, Matrix& out) {
    RLOG(LL_WARN, "grad_softmax not yet implemented");
}

// ─── Grad Step ────────────────────────────────────────────────────────────────

Matrix grad_step(const Matrix& x, const Matrix& dy) {
    RLOG(LL_WARN, "grad_step not yet implemented (gradient is zero)");
    return Matrix(x.rows(), x.cols());
}

void grad_step(const Matrix& x, const Matrix& dy, Matrix& out) {
    RLOG(LL_WARN, "grad_step not yet implemented (gradient is zero)");
}

// ─── Grad Threshold ───────────────────────────────────────────────────────────

Matrix grad_threshold(const Matrix& x, const Matrix& dy) {
    RLOG(LL_WARN, "grad_threshold not yet implemented (gradient is zero)");
    return Matrix(x.rows(), x.cols());
}

void grad_threshold(const Matrix& x, const Matrix& dy, Matrix& out) {
    RLOG(LL_WARN, "grad_threshold not yet implemented (gradient is zero)");
}
