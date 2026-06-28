#pragma once

#include <common.h>
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
// Template bodies are not instantiated until call time, so forward declarations
// above are sufficient — all types are complete by then.
//
// Lifetime note: expression types hold const-refs to their operands. Expressions
// must be evaluated (assigned to a Matrix) within the same full-expression that
// created them. Storing an expression in `auto` and using it later is UB when
// the operands are themselves temporary expressions.
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

    Matrix(int rows, int cols);  // TODO: cudaMalloc
    ~Matrix();                   // TODO: cudaFree

    Matrix(const Matrix&)            = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) noexcept;
    Matrix& operator=(Matrix&&) noexcept;

    i32 rows() const { return _rows; }
    i32 cols() const { return _cols; }

    // Evaluate an expression into this matrix's existing allocation.
    // Precondition: expr.rows() == rows() && expr.cols() == cols()
    template <typename Expr>
        requires (!std::is_same_v<std::decay_t<Expr>, Matrix>)
    Matrix& operator=(const Expr& expr);  // TODO: implement

    Matrix eval() const;  // TODO: deep GPU copy
};

// ─── Expression types ─────────────────────────────────────────────────────────

template <typename LHS, typename RHS>
struct MatrixMulExpr : MatrixExpr<MatrixMulExpr<LHS, RHS>> {
    const LHS& lhs;
    const RHS& rhs;
    i32 rows() const { return lhs.rows(); }
    i32 cols() const { return rhs.cols(); }
    Matrix eval() const;  // TODO: implement GEMM kernel
};

template <typename LHS, typename RHS>
struct MatrixAddExpr : MatrixExpr<MatrixAddExpr<LHS, RHS>> {
    const LHS& lhs;
    const RHS& rhs;
    i32 rows() const { return lhs.rows(); }
    i32 cols() const { return lhs.cols(); }
    Matrix eval() const;  // TODO: implement element-wise add kernel
};

template <typename LHS, typename RHS>
struct MatrixSubExpr : MatrixExpr<MatrixSubExpr<LHS, RHS>> {
    const LHS& lhs;
    const RHS& rhs;
    i32 rows() const { return lhs.rows(); }
    i32 cols() const { return lhs.cols(); }
    Matrix eval() const;  // TODO: implement element-wise sub kernel
};

template <typename LHS, typename RHS>
struct MatrixHadamardExpr : MatrixExpr<MatrixHadamardExpr<LHS, RHS>> {
    const LHS& lhs;
    const RHS& rhs;
    i32 rows() const { return lhs.rows(); }
    i32 cols() const { return lhs.cols(); }
    Matrix eval() const;  // TODO: implement element-wise mul kernel
};

template <typename LHS>
struct MatrixTransposeExpr : MatrixExpr<MatrixTransposeExpr<LHS>> {
    const LHS& lhs;
    i32 rows() const { return lhs.cols(); }  // dimensions flip
    i32 cols() const { return lhs.rows(); }
    Matrix eval() const;  // TODO: implement transpose kernel
};

template <typename LHS>
struct MatrixScalarMulExpr : MatrixExpr<MatrixScalarMulExpr<LHS>> {
    const LHS& lhs;
    f32        scalar;
    i32 rows() const { return lhs.rows(); }
    i32 cols() const { return lhs.cols(); }
    Matrix eval() const;  // TODO: implement scalar scale kernel
};

// col is a column vector (rows x 1); broadcast-added to every column of lhs
template <typename LHS, typename RHS>
struct MatrixColAddExpr : MatrixExpr<MatrixColAddExpr<LHS, RHS>> {
    const LHS& lhs;
    const RHS& col;
    i32 rows() const { return lhs.rows(); }
    i32 cols() const { return lhs.cols(); }
    Matrix eval() const;  // TODO: implement broadcast add kernel
};

// row is a row vector (1 x cols); broadcast-added to every row of lhs
template <typename LHS, typename RHS>
struct MatrixRowAddExpr : MatrixExpr<MatrixRowAddExpr<LHS, RHS>> {
    const LHS& lhs;
    const RHS& row;
    i32 rows() const { return lhs.rows(); }
    i32 cols() const { return lhs.cols(); }
    Matrix eval() const;  // TODO: implement broadcast add kernel
};
