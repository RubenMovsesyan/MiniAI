#pragma once

#include <matrix/matrix.cuh>

// ─── Activation expression types (for lazy/fused path) ─────────────────────────
// Each activation stores its input by value (like matrix ops); when invoked in an
// expression tree, __device__ operator() computes the function element-wise at
// compile time (inlined by the kernel). Lazy form: x.lazy().relu().eval().

template <typename LHS> struct ReluExpr : MatrixExpr<ReluExpr<LHS>> {
    LHS lhs;
    __host__ __device__ ReluExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return fmaxf(0.0f, lhs(r, c)); }
};

template <typename LHS> struct SigmoidExpr : MatrixExpr<SigmoidExpr<LHS>> {
    LHS lhs;
    __host__ __device__ SigmoidExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return 1.0f / (1.0f + expf(-lhs(r, c))); }
};

template <typename LHS> struct BipolarSigmoidExpr : MatrixExpr<BipolarSigmoidExpr<LHS>> {
    LHS lhs;
    __host__ __device__ BipolarSigmoidExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 x = lhs(r, c); return 2.0f / (1.0f + expf(-2.0f * x)) - 1.0f;
    }
};

template <typename LHS> struct TanhExpr : MatrixExpr<TanhExpr<LHS>> {
    LHS lhs;
    __host__ __device__ TanhExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return tanhf(lhs(r, c)); }
};

template <typename LHS> struct LeakyReluExpr : MatrixExpr<LeakyReluExpr<LHS>> {
    LHS lhs; f32 alpha;
    __host__ __device__ LeakyReluExpr(const LHS& l, f32 a = 0.01f) : lhs(l), alpha(a) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 x = lhs(r, c); return x > 0.0f ? x : alpha * x;
    }
};

template <typename LHS> struct SoftmaxExpr : MatrixExpr<SoftmaxExpr<LHS>> {
    LHS lhs;
    __host__ __device__ SoftmaxExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    // Note: row-wise softmax can't be computed element-wise in isolation.
    // This is a placeholder; actual softmax requires a kernel.
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c); }
};

template <typename LHS> struct StepExpr : MatrixExpr<StepExpr<LHS>> {
    LHS lhs; f32 threshold;
    __host__ __device__ StepExpr(const LHS& l, f32 t = 0.0f) : lhs(l), threshold(t) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) > threshold ? 1.0f : 0.0f; }
};

template <typename LHS> struct ThresholdExpr : MatrixExpr<ThresholdExpr<LHS>> {
    LHS lhs; f32 thresh;
    __host__ __device__ ThresholdExpr(const LHS& l, f32 t) : lhs(l), thresh(t) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) > thresh ? 1.0f : 0.0f; }
};

// ─── Eager forward declarations ────────────────────────────────────────────────
// Each has owned-return and out-param overload.
Matrix relu(const Matrix& x);
void   relu(const Matrix& x, Matrix& out);

Matrix sigmoid(const Matrix& x);
void   sigmoid(const Matrix& x, Matrix& out);

Matrix bipolar_sigmoid(const Matrix& x);
void   bipolar_sigmoid(const Matrix& x, Matrix& out);

Matrix tanh(const Matrix& x);
void   tanh(const Matrix& x, Matrix& out);

Matrix leaky_relu(const Matrix& x, f32 alpha = 0.01f);
void   leaky_relu(const Matrix& x, f32 alpha, Matrix& out);

Matrix softmax(const Matrix& x);
void   softmax(const Matrix& x, Matrix& out);

Matrix step(const Matrix& x, f32 threshold = 0.0f);
void   step(const Matrix& x, f32 threshold, Matrix& out);

Matrix threshold(const Matrix& x, f32 thresh);
void   threshold(const Matrix& x, f32 thresh, Matrix& out);
