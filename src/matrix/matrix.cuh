#pragma once

#include <common.h>
#include <cuda_runtime.h>
#include <type_traits>

// Forward declarations — needed so MatrixExpr can name return types before full definitions
template <typename LHS, typename RHS> struct MatrixMulExpr;
template <typename LHS, typename RHS> struct MatrixAddExpr;
template <typename LHS, typename RHS> struct MatrixSubExpr;
template <typename LHS, typename RHS> struct MatrixHadamardExpr;
template <typename LHS>               struct MatrixTransposeExpr;
template <typename LHS>               struct MatrixScalarMulExpr;
template <typename LHS, typename RHS> struct MatrixColAddExpr;
template <typename LHS, typename RHS> struct MatrixRowAddExpr;

class Matrix;

// ─── CRTP base ────────────────────────────────────────────────────────────────
// Gives Matrix and every expression type the same operator overloads.
// Operators build expression trees on the host; no GPU work happens here.
// Template bodies are not instantiated until call time, so forward declarations
// above are sufficient — all types are complete by then.
template <typename Derived>
struct MatrixExpr {
    const Derived& self() const { return static_cast<const Derived&>(*this); }

    template <typename RHS>
    MatrixMulExpr<Derived, RHS> operator*(const RHS& rhs) const { return {self(), rhs}; }

    // Non-template overload wins over the template above when rhs is f32
    MatrixScalarMulExpr<Derived> operator*(f32 s) const { return {self(), s}; }

    template <typename RHS>
    MatrixAddExpr<Derived, RHS> operator+(const RHS& rhs) const { return {self(), rhs}; }

    template <typename RHS>
    MatrixSubExpr<Derived, RHS> operator-(const RHS& rhs) const { return {self(), rhs}; }

    template <typename RHS>
    MatrixHadamardExpr<Derived, RHS> hadamard(const RHS& rhs) const { return {self(), rhs}; }

    MatrixTransposeExpr<Derived> transpose() const { return {self()}; }

    // col: add a column vector (rows x 1) to every column of lhs
    template <typename RHS>
    MatrixColAddExpr<Derived, RHS> colAdd(const RHS& col) const { return {self(), col}; }

    // row: add a row vector (1 x cols) to every row of lhs
    template <typename RHS>
    MatrixRowAddExpr<Derived, RHS> rowAdd(const RHS& row) const { return {self(), row}; }
};

// ─── Matrix ───────────────────────────────────────────────────────────────────
class Matrix : public MatrixExpr<Matrix> {
public:
    f32* data;   // device pointer (GPU memory)
    i32  _rows, _cols;

    Matrix(i32 rows, i32 cols);  // cudaMalloc
    ~Matrix();                   // cudaFree

    Matrix(const Matrix&)            = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) noexcept;
    Matrix& operator=(Matrix&&) noexcept;

    __host__ __device__ i32 rows() const { return _rows; }
    __host__ __device__ i32 cols() const { return _cols; }

    // Element access — called by the eval kernel and host-side utilities
    __host__ __device__ f32 operator()(i32 r, i32 c) const { return data[r * _cols + c]; }

    // Evaluate any expression into this matrix's existing allocation via a fused kernel.
    // Precondition: expr.rows() == rows() && expr.cols() == cols()
    template <typename Expr>
        requires (!std::is_same_v<std::decay_t<Expr>, Matrix>)
    Matrix& operator=(const Expr& expr);

    Matrix eval() const;  // deep GPU copy (see matrix.cu)
};

// ─── Evaluation kernel ────────────────────────────────────────────────────────
// One thread per output element; calls expr(r, c) which recursively evaluates
// the full expression tree at compile time with no intermediate allocations.
// Expression types store operands by value so no host pointers are in device code.
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

// ─── Expression types ─────────────────────────────────────────────────────────
// Operands are stored by value so the expression tree is self-contained when
// passed to a CUDA kernel — no host-address references in device code.

// MatrixMulExpr has no __device__ operator() — matrix multiply requires a
// dedicated GEMM kernel, not the element-wise eval path. Implement later.
template <typename LHS, typename RHS>
struct MatrixMulExpr : MatrixExpr<MatrixMulExpr<LHS, RHS>> {
    LHS lhs;
    RHS rhs;
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return rhs.cols(); }
};

template <typename LHS, typename RHS>
struct MatrixAddExpr : MatrixExpr<MatrixAddExpr<LHS, RHS>> {
    LHS lhs;
    RHS rhs;
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) + rhs(r, c); }
};

template <typename LHS, typename RHS>
struct MatrixSubExpr : MatrixExpr<MatrixSubExpr<LHS, RHS>> {
    LHS lhs;
    RHS rhs;
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) - rhs(r, c); }
};

template <typename LHS, typename RHS>
struct MatrixHadamardExpr : MatrixExpr<MatrixHadamardExpr<LHS, RHS>> {
    LHS lhs;
    RHS rhs;
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) * rhs(r, c); }
};

template <typename LHS>
struct MatrixTransposeExpr : MatrixExpr<MatrixTransposeExpr<LHS>> {
    LHS lhs;
    __host__ __device__ i32 rows() const { return lhs.cols(); }  // dimensions flip
    __host__ __device__ i32 cols() const { return lhs.rows(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(c, r); }
};

template <typename LHS>
struct MatrixScalarMulExpr : MatrixExpr<MatrixScalarMulExpr<LHS>> {
    LHS lhs;
    f32 scalar;
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) * scalar; }
};

// col is a column vector (rows x 1); broadcast-added to every column of lhs
template <typename LHS, typename RHS>
struct MatrixColAddExpr : MatrixExpr<MatrixColAddExpr<LHS, RHS>> {
    LHS lhs;
    RHS col;
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) + col(r, 0); }
};

// row is a row vector (1 x cols); broadcast-added to every row of lhs
template <typename LHS, typename RHS>
struct MatrixRowAddExpr : MatrixExpr<MatrixRowAddExpr<LHS, RHS>> {
    LHS lhs;
    RHS row;
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const { return lhs(r, c) + row(0, c); }
};
