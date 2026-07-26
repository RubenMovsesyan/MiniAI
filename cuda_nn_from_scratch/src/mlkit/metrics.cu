#include <rlog.h>
#include <mlkit/metrics.cuh>
#include <agg/agg.cuh>

// One thread per row; bump the device counter when the predicted class matches the true one.
__global__ static void countEqualKernel(const f32* a, const f32* b, i32* counter, i32 n) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] == b[i]) atomicAdd(counter, 1);
}

AccuracyMeter::AccuracyMeter(i32 batch) : pred_(batch, 1), truth_(batch, 1) {
    cudaMalloc(&d_correct_, sizeof(i32));
    reset();
}

AccuracyMeter::~AccuracyMeter() {
    if (d_correct_) cudaFree(d_correct_);
}

void AccuracyMeter::reset() {
    cudaMemsetAsync(d_correct_, 0, sizeof(i32), g_compute_stream);
    total_ = 0;
}

void AccuracyMeter::update(const Matrix& logits, const Matrix& targets) {
    i32 n = logits.rows();
    row_argmax(logits, pred_);     // predicted class per row
    row_argmax(targets, truth_);   // targets are one-hot → argmax is the true class
    i32 threads = 256;
    i32 blocks  = (n + threads - 1) / threads;
    countEqualKernel<<<blocks, threads, 0, g_compute_stream>>>(
        pred_.data, truth_.data, d_correct_, n);
    total_ += n;
}

f32 AccuracyMeter::value() {
    if (total_ == 0) return 0.0f;
    cudaStreamSynchronize(g_compute_stream);   // the one readback for the whole pass
    i32 correct = 0;
    cudaMemcpy(&correct, d_correct_, sizeof(i32), cudaMemcpyDeviceToHost);
    return (f32)correct / (f32)total_;
}
