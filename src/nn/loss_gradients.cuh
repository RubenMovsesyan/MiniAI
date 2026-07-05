#pragma once

#include <matrix/matrix.cuh>

// ─── Loss gradient declarations (stubs for future implementation) ───────────────

Matrix grad_mse(const Matrix& pred, const Matrix& target);
Matrix grad_cross_entropy(const Matrix& logits, const Matrix& targets);
Matrix grad_l1_loss(const Matrix& pred, const Matrix& target);
Matrix grad_l2_loss(const Matrix& pred, const Matrix& target);
