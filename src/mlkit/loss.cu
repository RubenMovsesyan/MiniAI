#include <rlog.h>
#include <mlkit/loss.cuh>
#include <nn/losses.cuh>
#include <fused/fused.cuh>

const Matrix& SoftmaxCrossEntropyLoss::backward(const Matrix& logits, const Matrix& targets) {
    // (a2 − y)/N into preallocated buffers — no allocation, so nothing syncs the stream.
    grad_softmax_cross_entropy(logits, targets, grad, a2);
    return grad;
}

f32 SoftmaxCrossEntropyLoss::value(const Matrix& logits, const Matrix& targets) {
    Matrix loss = cross_entropy(logits, targets);        // Matrix(1,1) on device
    cudaStreamSynchronize(g_compute_stream);             // the deliberate readback point
    f32 h;
    cudaMemcpy(&h, loss.data, sizeof(f32), cudaMemcpyDeviceToHost);
    return h;
}
