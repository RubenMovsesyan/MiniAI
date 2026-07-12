#pragma once

#include <matrix/matrix.cuh>

// ─── AccuracyMeter ──────────────────────────────────────────────────────────────
// Classification accuracy over an evaluation pass. Accuracy spans many batches, so
// this accumulates rather than returning a per-batch scalar — and it counts on-device
// (atomicAdd into a counter), so update() never syncs. The whole pass costs exactly
// ONE host readback, in value().
//
//   AccuracyMeter meter(B);
//   meter.reset();
//   for (batch) meter.update(net.forward(Xb), Yb);
//   f32 acc = meter.value();          // fraction in [0, 1]
//
// A row counts as correct when argmax(logits) == argmax(targets) — targets are one-hot,
// so their argmax is the true class.

class AccuracyMeter {
public:
    explicit AccuracyMeter(i32 batch);
    ~AccuracyMeter();

    AccuracyMeter(const AccuracyMeter&)            = delete;
    AccuracyMeter& operator=(const AccuracyMeter&) = delete;

    void reset();                                              // zero the counter + total
    void update(const Matrix& logits, const Matrix& targets);  // async, no sync
    f32  value();                                              // syncs once; correct / total

    i64 total() const { return total_; }   // samples counted so far

private:
    Matrix pred_, truth_;        // (B × 1) argmax outputs
    i32*   d_correct_ = nullptr; // device counter, allocated once
    i64    total_ = 0;
};
