#pragma once

#include <matrix/matrix.cuh>

// ─── Gradient expression types ─────────────────────────────────────────────────
// Each gradient takes the forward activation output (or input if needed) and
// the upstream gradient (dy), and computes the local gradient (dx).

template <typename LHS, typename RHS> struct GradReluExpr : MatrixExpr<GradReluExpr<LHS, RHS>> {
    LHS x_fwd;  // forward input
    RHS dy;     // upstream gradient
    __host__ __device__ GradReluExpr(const LHS& x, const RHS& d) : x_fwd(x), dy(d) {}
    __host__ __device__ i32 rows() const { return x_fwd.rows(); }
    __host__ __device__ i32 cols() const { return x_fwd.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        return x_fwd(r, c) > 0.0f ? dy(r, c) : 0.0f;
    }
};

template <typename LHS, typename RHS> struct GradSigmoidExpr : MatrixExpr<GradSigmoidExpr<LHS, RHS>> {
    LHS y_out;  // sigmoid output (not input)
    RHS dy;
    __host__ __device__ GradSigmoidExpr(const LHS& y, const RHS& d) : y_out(y), dy(d) {}
    __host__ __device__ i32 rows() const { return y_out.rows(); }
    __host__ __device__ i32 cols() const { return y_out.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 y = y_out(r, c); return dy(r, c) * y * (1.0f - y);
    }
};

template <typename LHS, typename RHS> struct GradBipolarSigmoidExpr : MatrixExpr<GradBipolarSigmoidExpr<LHS, RHS>> {
    LHS y_out;
    RHS dy;
    __host__ __device__ GradBipolarSigmoidExpr(const LHS& y, const RHS& d) : y_out(y), dy(d) {}
    __host__ __device__ i32 rows() const { return y_out.rows(); }
    __host__ __device__ i32 cols() const { return y_out.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 y = y_out(r, c); return dy(r, c) * (1.0f - y * y);  // 2x standard sigmoid deriv
    }
};

template <typename LHS, typename RHS> struct GradTanhExpr : MatrixExpr<GradTanhExpr<LHS, RHS>> {
    LHS y_out;
    RHS dy;
    __host__ __device__ GradTanhExpr(const LHS& y, const RHS& d) : y_out(y), dy(d) {}
    __host__ __device__ i32 rows() const { return y_out.rows(); }
    __host__ __device__ i32 cols() const { return y_out.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 y = y_out(r, c); return dy(r, c) * (1.0f - y * y);
    }
};

template <typename LHS, typename RHS> struct GradLeakyReluExpr : MatrixExpr<GradLeakyReluExpr<LHS, RHS>> {
    LHS x_fwd;
    RHS dy;
    f32 alpha;
    __host__ __device__ GradLeakyReluExpr(const LHS& x, const RHS& d, f32 a = 0.01f)
        : x_fwd(x), dy(d), alpha(a) {}
    __host__ __device__ i32 rows() const { return x_fwd.rows(); }
    __host__ __device__ i32 cols() const { return x_fwd.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        return x_fwd(r, c) > 0.0f ? dy(r, c) : alpha * dy(r, c);
    }
};

template <typename LHS, typename RHS> struct GradSoftmaxExpr : MatrixExpr<GradSoftmaxExpr<LHS, RHS>> {
    LHS y_out;
    RHS dy;
    __host__ __device__ GradSoftmaxExpr(const LHS& y, const RHS& d) : y_out(y), dy(d) {}
    __host__ __device__ i32 rows() const { return y_out.rows(); }
    __host__ __device__ i32 cols() const { return y_out.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return dy(r, c); }  // placeholder
};

template <typename LHS, typename RHS> struct GradStepExpr : MatrixExpr<GradStepExpr<LHS, RHS>> {
    LHS x_fwd;
    RHS dy;
    __host__ __device__ GradStepExpr(const LHS& x, const RHS& d) : x_fwd(x), dy(d) {}
    __host__ __device__ i32 rows() const { return x_fwd.rows(); }
    __host__ __device__ i32 cols() const { return x_fwd.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return 0.0f; }  // step gradient is zero almost everywhere
};

template <typename LHS, typename RHS> struct GradThresholdExpr : MatrixExpr<GradThresholdExpr<LHS, RHS>> {
    LHS x_fwd;
    RHS dy;
    __host__ __device__ GradThresholdExpr(const LHS& x, const RHS& d) : x_fwd(x), dy(d) {}
    __host__ __device__ i32 rows() const { return x_fwd.rows(); }
    __host__ __device__ i32 cols() const { return x_fwd.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return 0.0f; }  // threshold gradient is zero
};

// ─── Eager gradient declarations ───────────────────────────────────────────────
// grad_XXX(x_fwd_or_y_out, dy) → dx (where dy is upstream gradient)

Matrix grad_relu(const Matrix& x, const Matrix& dy);
void   grad_relu(const Matrix& x, const Matrix& dy, Matrix& out);

Matrix grad_sigmoid(const Matrix& y, const Matrix& dy);
void   grad_sigmoid(const Matrix& y, const Matrix& dy, Matrix& out);

Matrix grad_bipolar_sigmoid(const Matrix& y, const Matrix& dy);
void   grad_bipolar_sigmoid(const Matrix& y, const Matrix& dy, Matrix& out);

Matrix grad_tanh(const Matrix& y, const Matrix& dy);
void   grad_tanh(const Matrix& y, const Matrix& dy, Matrix& out);

Matrix grad_leaky_relu(const Matrix& x, const Matrix& dy, f32 alpha = 0.01f);
void   grad_leaky_relu(const Matrix& x, const Matrix& dy, f32 alpha, Matrix& out);

Matrix grad_softmax(const Matrix& y, const Matrix& dy);
void   grad_softmax(const Matrix& y, const Matrix& dy, Matrix& out);

Matrix grad_step(const Matrix& x, const Matrix& dy);
void   grad_step(const Matrix& x, const Matrix& dy, Matrix& out);

Matrix grad_threshold(const Matrix& x, const Matrix& dy);
void   grad_threshold(const Matrix& x, const Matrix& dy, Matrix& out);
