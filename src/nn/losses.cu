#include <rlog.h>
#include <nn/losses.cuh>

Matrix mse(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "mse not yet implemented");
    return Matrix(1, 1);
}

Matrix cross_entropy(const Matrix& logits, const Matrix& targets) {
    RLOG(LL_WARN, "cross_entropy not yet implemented");
    return Matrix(1, 1);
}

Matrix l1_loss(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "l1_loss not yet implemented");
    return Matrix(1, 1);
}

Matrix l2_loss(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "l2_loss not yet implemented");
    return Matrix(1, 1);
}
