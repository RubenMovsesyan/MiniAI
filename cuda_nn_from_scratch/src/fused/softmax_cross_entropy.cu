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
    Matrix a2(logits.rows(), logits.cols());
    grad_softmax_cross_entropy(logits, targets, out, a2);
}

void grad_softmax_cross_entropy(const Matrix& logits, const Matrix& targets,
                                Matrix& out, Matrix& a2) {
    i32 N = logits.rows();
    softmax(logits, a2);                                   // out-param: no allocation
    out = (a2.ref() - targets.ref()) * (1.0f / (f32)N);    // one fused kernel
}
