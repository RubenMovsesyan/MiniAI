#pragma once

#include <matrix/matrix.cuh>

// ─── Fused softmax⊕cross-entropy gradient ──────────────────────────────────────
// The softmax Jacobian cancels against the cross-entropy derivative, so the
// gradient of CE w.r.t. the logits collapses to (softmax(logits) - targets) / N,
// where targets is a one-hot matrix (rows×classes) and N = rows. Pure element-wise.

Matrix grad_softmax_cross_entropy(const Matrix& logits, const Matrix& targets);
void   grad_softmax_cross_entropy(const Matrix& logits, const Matrix& targets, Matrix& out);

// Allocation-free form: the caller supplies the softmax scratch buffer (B×classes).
// Use this on the training hot path — cudaMalloc/cudaFree are device-synchronizing,
// so an internally-allocated a2 would stall the async stream every step.
void   grad_softmax_cross_entropy(const Matrix& logits, const Matrix& targets,
                                  Matrix& out, Matrix& a2);
