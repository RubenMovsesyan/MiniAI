#pragma once

#include <matrix/matrix.cuh>

// ─── Loss function declarations (stubs for future implementation) ──────────────

Matrix mse(const Matrix& pred, const Matrix& target);
Matrix cross_entropy(const Matrix& logits, const Matrix& targets);
Matrix l1_loss(const Matrix& pred, const Matrix& target);
Matrix l2_loss(const Matrix& pred, const Matrix& target);
