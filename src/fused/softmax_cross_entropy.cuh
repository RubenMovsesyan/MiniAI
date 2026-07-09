#pragma once

#include <matrix/matrix.cuh>

// ─── Fused softmax⊕cross-entropy gradient ──────────────────────────────────────
// The softmax Jacobian cancels against the cross-entropy derivative, so the
// gradient of CE w.r.t. the logits collapses to (softmax(logits) - targets) / N,
// where targets is a one-hot matrix (rows×classes) and N = rows. Pure element-wise.

Matrix grad_softmax_cross_entropy(const Matrix& logits, const Matrix& targets);
void   grad_softmax_cross_entropy(const Matrix& logits, const Matrix& targets, Matrix& out);
