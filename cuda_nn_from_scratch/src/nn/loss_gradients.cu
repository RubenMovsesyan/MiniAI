#include <rlog.h>
#include <nn/loss_gradients.cuh>

Matrix grad_mse(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "grad_mse not yet implemented");
    return Matrix(pred.rows(), pred.cols());
}

Matrix grad_l1_loss(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "grad_l1_loss not yet implemented");
    return Matrix(pred.rows(), pred.cols());
}

Matrix grad_l2_loss(const Matrix& pred, const Matrix& target) {
    RLOG(LL_WARN, "grad_l2_loss not yet implemented");
    return Matrix(pred.rows(), pred.cols());
}
