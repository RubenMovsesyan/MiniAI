#pragma once

#include <matrix/matrix.cuh>

// ─── Loss gradient declarations (stubs for future implementation) ───────────────

Matrix grad_mse(const Matrix& pred, const Matrix& target);
// grad_cross_entropy is fused with softmax — see grad_softmax_cross_entropy in src/fused/.
Matrix grad_l1_loss(const Matrix& pred, const Matrix& target);
Matrix grad_l2_loss(const Matrix& pred, const Matrix& target);
