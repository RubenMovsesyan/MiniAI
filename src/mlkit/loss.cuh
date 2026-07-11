#pragma once

#include <matrix/matrix.cuh>

// ─── Softmax cross-entropy loss ─────────────────────────────────────────────────
// NOT a layer. The final Linear layer emits raw logits (identity activation); this
// object owns the softmax. backward() injects (a2 − y)/N as the gradient handed to
// that last layer — the softmax Jacobian cancels, so softmax appears exactly once,
// here, and is never backward-chained as an activation.

struct SoftmaxCrossEntropyLoss {
    Matrix grad;   // (B × classes) — reused each step, holds (a2 − y)/N

    SoftmaxCrossEntropyLoss(i32 batch, i32 classes) : grad(batch, classes) {}

    // Gradient w.r.t. logits, every step. targets are one-hot (B × classes).
    const Matrix& backward(const Matrix& logits, const Matrix& targets);

    // Scalar loss value — the one host readback point (syncs the stream). Called only
    // on reporting steps, so recomputing softmax here is cheap and rare.
    f32 value(const Matrix& logits, const Matrix& targets);
};
