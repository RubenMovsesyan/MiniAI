#pragma once

#include <common.h>
#include <cuda_runtime.h>
#include <type_traits>

// ─── Forward declarations ─────────────────────────────────────────────────────
template <typename LHS, typename RHS> struct MatrixMulExpr;
template <typename LHS, typename RHS> struct MatrixAddExpr;
template <typename LHS, typename RHS> struct MatrixSubExpr;
template <typename LHS, typename RHS> struct MatrixHadamardExpr;
template <typename LHS>               struct MatrixTransposeExpr;
template <typename LHS>               struct MatrixScalarMulExpr;
template <typename LHS, typename RHS> struct MatrixColAddExpr;
template <typename LHS, typename RHS> struct MatrixRowAddExpr;
struct MatrixRef;
class Matrix;

// ─── nodeOf ───────────────────────────────────────────────────────────────────
// Converts an operand to the type safe for value-storage in expression nodes.
// Matrix (non-copyable owner) → MatrixRef view; everything else → pass through.
// The Matrix overload is declared here and defined inline after Matrix below.
MatrixRef nodeOf(const Matrix& m);
template <typename T> T nodeOf(const T& t) { return t; }

// Storage type used in expression nodes for a given operand type
template <typename T>
using NodeOf_t = decltype(nodeOf(std::declval<const T&>()));

// ─── CRTP base — method declarations only ────────────────────────────────────
// Derived is MatrixRef or one of the expression structs — never Matrix.
// Bodies are defined after all expression types are complete (see below).
template <typename Derived>
struct MatrixExpr {
    const Derived& self() const { return static_cast<const Derived&>(*this); }

    template <typename RHS>
    MatrixMulExpr<Derived, NodeOf_t<RHS>>      operator*(const RHS& rhs) const;
    MatrixScalarMulExpr<Derived>               operator*(f32 s)          const;
    template <typename RHS>
    MatrixAddExpr<Derived, NodeOf_t<RHS>>      operator+(const RHS& rhs) const;
    template <typename RHS>
    MatrixSubExpr<Derived, NodeOf_t<RHS>>      operator-(const RHS& rhs) const;
    template <typename RHS>
    MatrixHadamardExpr<Derived, NodeOf_t<RHS>> hadamard(const RHS& rhs)  const;
    MatrixTransposeExpr<Derived>               transpose()               const;
    template <typename RHS>
    MatrixColAddExpr<Derived, NodeOf_t<RHS>>   colAdd(const RHS& col)    const;
    template <typename RHS>
    MatrixRowAddExpr<Derived, NodeOf_t<RHS>>   rowAdd(const RHS& row)    const;
};

// ─── MatrixRef ────────────────────────────────────────────────────────────────
// Trivially-copyable view into GPU data — the leaf type in every expression tree.
// Safe to pass to CUDA kernels by value.
struct MatrixRef : MatrixExpr<MatrixRef> {
    f32* data;
    i32  _rows, _cols;

    __host__ __device__ MatrixRef(f32* d, i32 r, i32 c) : data(d), _rows(r), _cols(c) {}
    __host__ __device__ i32 rows() const { return _rows; }
    __host__ __device__ i32 cols() const { return _cols; }
    __host__ __device__ f32 operator()(i32 r, i32 c) const { return data[r * _cols + c]; }
};

// ─── Matrix ───────────────────────────────────────────────────────────────────
// Owns GPU memory. Not a MatrixExpr subtype — use ref() / implicit conversion
// to build expression trees. Provides forwarding operators for ergonomics.
// Forwarding operator bodies are defined after expression types are complete.
class Matrix {
public:
    f32* data;
    i32  _rows, _cols;

    Matrix(i32 rows, i32 cols);
    ~Matrix();
    Matrix(const Matrix&)            = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) noexcept;
    Matrix& operator=(Matrix&&) noexcept;

    __host__ __device__ i32 rows() const { return _rows; }
    __host__ __device__ i32 cols() const { return _cols; }
    __host__ __device__ f32 operator()(i32 r, i32 c) const { return data[r * _cols + c]; }

    MatrixRef ref()       const { return {data, _rows, _cols}; }
    operator  MatrixRef() const { return ref(); }

    // Forwarding operators — all go through ref() so expression trees use MatrixRef leaves.
    // Bodies are defined out-of-class after expression types are fully defined.
    template <typename RHS> auto operator*(const RHS& rhs) const;
    auto                         operator*(f32 s)          const;
    template <typename RHS> auto operator+(const RHS& rhs) const;
    template <typename RHS> auto operator-(const RHS& rhs) const;
    template <typename RHS> auto hadamard(const RHS& rhs)  const;
    auto                         transpose()               const;
    template <typename RHS> auto colAdd(const RHS& col)    const;
    template <typename RHS> auto rowAdd(const RHS& row)    const;

    // Evaluate any expression into this matrix's existing allocation via a fused kernel.
    // Precondition: expr.rows() == rows() && expr.cols() == cols()
    template <typename Expr>
        requires (!std::is_same_v<std::decay_t<Expr>, Matrix>)
    Matrix& operator=(const Expr& expr);

    Matrix eval() const;
};

// nodeOf(Matrix) definition — now that Matrix is complete
inline MatrixRef nodeOf(const Matrix& m) { return m.ref(); }

// ─── Expression types ─────────────────────────────────────────────────────────
// Operands stored by value. LHS/RHS are MatrixRef or nested expression types —
// never Matrix. The nodeOf() conversions in MatrixExpr operators ensure this.

// Stub until a real GEMM kernel is wired in. Always returns 0.0f so that
// matmul tests compile and run but fail the correctness check.
template <typename LHS, typename RHS>
struct MatrixMulExpr : MatrixExpr<MatrixMulExpr<LHS, RHS>> {
    LHS lhs; RHS rhs;
    __host__ __device__ MatrixMulExpr(const LHS& l, const RHS& r) : lhs(l), rhs(r) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return rhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return 0.0f; }
};

template <typename LHS, typename RHS>
struct MatrixAddExpr : MatrixExpr<MatrixAddExpr<LHS, RHS>> {
    LHS lhs; RHS rhs;
    __host__ __device__ MatrixAddExpr(const LHS& l, const RHS& r) : lhs(l), rhs(r) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) + rhs(r, c); }
};

template <typename LHS, typename RHS>
struct MatrixSubExpr : MatrixExpr<MatrixSubExpr<LHS, RHS>> {
    LHS lhs; RHS rhs;
    __host__ __device__ MatrixSubExpr(const LHS& l, const RHS& r) : lhs(l), rhs(r) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) - rhs(r, c); }
};

template <typename LHS, typename RHS>
struct MatrixHadamardExpr : MatrixExpr<MatrixHadamardExpr<LHS, RHS>> {
    LHS lhs; RHS rhs;
    __host__ __device__ MatrixHadamardExpr(const LHS& l, const RHS& r) : lhs(l), rhs(r) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) * rhs(r, c); }
};

template <typename LHS>
struct MatrixTransposeExpr : MatrixExpr<MatrixTransposeExpr<LHS>> {
    LHS lhs;
    __host__ __device__ MatrixTransposeExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return lhs.cols(); }
    __host__ __device__ i32 cols() const { return lhs.rows(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(c, r); }
};

template <typename LHS>
struct MatrixScalarMulExpr : MatrixExpr<MatrixScalarMulExpr<LHS>> {
    LHS lhs; f32 scalar;
    __host__ __device__ MatrixScalarMulExpr(const LHS& l, f32 s) : lhs(l), scalar(s) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) * scalar; }
};

// col is a column vector (rows x 1); broadcast-added to every column of lhs
template <typename LHS, typename RHS>
struct MatrixColAddExpr : MatrixExpr<MatrixColAddExpr<LHS, RHS>> {
    LHS lhs; RHS col;
    __host__ __device__ MatrixColAddExpr(const LHS& l, const RHS& c) : lhs(l), col(c) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) + col(r, 0); }
};

// row is a row vector (1 x cols); broadcast-added to every row of lhs
template <typename LHS, typename RHS>
struct MatrixRowAddExpr : MatrixExpr<MatrixRowAddExpr<LHS, RHS>> {
    LHS lhs; RHS row;
    __host__ __device__ MatrixRowAddExpr(const LHS& l, const RHS& r) : lhs(l), row(r) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) + row(0, c); }
};

// ─── MatrixExpr method definitions ───────────────────────────────────────────
// All expression types are now complete — aggregate init works.

template <typename Derived> template <typename RHS>
MatrixMulExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::operator*(const RHS& rhs) const { return {self(), nodeOf(rhs)}; }

template <typename Derived>
MatrixScalarMulExpr<Derived>
MatrixExpr<Derived>::operator*(f32 s) const { return {self(), s}; }

template <typename Derived> template <typename RHS>
MatrixAddExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::operator+(const RHS& rhs) const { return {self(), nodeOf(rhs)}; }

template <typename Derived> template <typename RHS>
MatrixSubExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::operator-(const RHS& rhs) const { return {self(), nodeOf(rhs)}; }

template <typename Derived> template <typename RHS>
MatrixHadamardExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::hadamard(const RHS& rhs) const { return {self(), nodeOf(rhs)}; }

template <typename Derived>
MatrixTransposeExpr<Derived>
MatrixExpr<Derived>::transpose() const { return {self()}; }

template <typename Derived> template <typename RHS>
MatrixColAddExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::colAdd(const RHS& col) const { return {self(), nodeOf(col)}; }

template <typename Derived> template <typename RHS>
MatrixRowAddExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::rowAdd(const RHS& row) const { return {self(), nodeOf(row)}; }

// ─── Matrix forwarding operator definitions ───────────────────────────────────

template <typename RHS>
auto Matrix::operator*(const RHS& rhs) const { return ref() * rhs; }
inline auto Matrix::operator*(f32 s) const    { return ref() * s; }
template <typename RHS>
auto Matrix::operator+(const RHS& rhs) const { return ref() + rhs; }
template <typename RHS>
auto Matrix::operator-(const RHS& rhs) const { return ref() - rhs; }
template <typename RHS>
auto Matrix::hadamard(const RHS& rhs) const  { return ref().hadamard(rhs); }
inline auto Matrix::transpose() const        { return ref().transpose(); }
template <typename RHS>
auto Matrix::colAdd(const RHS& col) const    { return ref().colAdd(col); }
template <typename RHS>
auto Matrix::rowAdd(const RHS& row) const    { return ref().rowAdd(row); }

// ─── Evaluation kernel ────────────────────────────────────────────────────────
// One thread per output element; calls expr(r, c) which recursively evaluates
// the full expression tree at compile time with no intermediate allocations.
template <typename Expr>
__global__ void matEvalKernel(Expr expr, f32* out, i32 rows, i32 cols) {
    i32 r = blockIdx.y * blockDim.y + threadIdx.y;
    i32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols)
        out[r * cols + c] = expr(r, c);
}

// Matrix::operator= template body must live in the header
template <typename Expr>
    requires (!std::is_same_v<std::decay_t<Expr>, Matrix>)
Matrix& Matrix::operator=(const Expr& expr) {
    dim3 block(16, 16);
    dim3 grid((_cols + 15) / 16, (_rows + 15) / 16);
    matEvalKernel<<<grid, block>>>(expr, data, _rows, _cols);
    cudaDeviceSynchronize();
    return *this;
}
