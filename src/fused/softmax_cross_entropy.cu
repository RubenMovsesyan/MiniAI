#include <rlog.h>
#include <fused/softmax_cross_entropy.cuh>
#include <nn/activations.cuh>

// grad = (softmax(logits) - targets) / N — one fused matEvalKernel (sub + scalar-mul).

Matrix grad_softmax_cross_entropy(const Matrix& logits, const Matrix& targets) {
    Matrix out(logits.rows(), logits.cols());
    grad_softmax_cross_entropy(logits, targets, out);
    return out;
}

void grad_softmax_cross_entropy(const Matrix& logits, const Matrix& targets, Matrix& out) {
    i32 N = logits.rows();
    Matrix a2 = softmax(logits);
    out = (a2 - targets) * (1.0f / (f32)N);
}
