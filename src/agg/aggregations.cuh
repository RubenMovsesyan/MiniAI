#pragma once

#include <matrix/matrix.cuh>

// ─── Row sum expression type (sums across columns) ──────────────────────────
// RowSumExpr(4×3) → expression with rows=4, cols=1
// Each __device__ operator()(r, c) sums row r across all columns

template <typename LHS> struct RowSumExpr : MatrixExpr<RowSumExpr<LHS>> {
    LHS lhs;
    __host__ __device__ RowSumExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return 1; }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 sum = 0.0f;
        for (i32 j = 0; j < lhs.cols(); j++)
            sum += lhs(r, j);
        return sum;
    }
};

// ─── Column sum expression type (sums across rows) ─────────────────────────
// ColSumExpr(4×3) → expression with rows=1, cols=3
// Each __device__ operator()(r, c) sums column c across all rows

template <typename LHS> struct ColSumExpr : MatrixExpr<ColSumExpr<LHS>> {
    LHS lhs;
    __host__ __device__ ColSumExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return 1; }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 sum = 0.0f;
        for (i32 i = 0; i < lhs.rows(); i++)
            sum += lhs(i, c);
        return sum;
    }
};

// ─── Row max expression type (max across columns) ───────────────────────────
// RowMaxExpr(4×3) → expression with rows=4, cols=1
// Each __device__ operator()(r, c) finds max of row r across all columns

template <typename LHS> struct RowMaxExpr : MatrixExpr<RowMaxExpr<LHS>> {
    LHS lhs;
    __host__ __device__ RowMaxExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return lhs.rows(); }
    __host__ __device__ i32 cols() const { return 1; }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 maxval = -INFINITY;
        for (i32 j = 0; j < lhs.cols(); j++)
            maxval = fmaxf(maxval, lhs(r, j));
        return maxval;
    }
};

// ─── Column max expression type (max across rows) ────────────────────────────
// ColMaxExpr(4×3) → expression with rows=1, cols=3
// Each __device__ operator()(r, c) finds max of column c across all rows

template <typename LHS> struct ColMaxExpr : MatrixExpr<ColMaxExpr<LHS>> {
    LHS lhs;
    __host__ __device__ ColMaxExpr(const LHS& l) : lhs(l) {}
    __host__ __device__ i32 rows() const { return 1; }
    __host__ __device__ i32 cols() const { return lhs.cols(); }
    __device__ f32 operator()(i32 r, i32 c) const {
        f32 maxval = -INFINITY;
        for (i32 i = 0; i < lhs.rows(); i++)
            maxval = fmaxf(maxval, lhs(i, c));
        return maxval;
    }
};

// ─── Eager forward declarations ────────────────────────────────────────────

// Row sum: Matrix(rows, cols) → Matrix(rows, 1)
Matrix row_sum(const Matrix& x);
void   row_sum(const Matrix& x, Matrix& out);

// Column sum: Matrix(rows, cols) → Matrix(1, cols)
Matrix col_sum(const Matrix& x);
void   col_sum(const Matrix& x, Matrix& out);

// Total sum: chains row_sum then col_sum → Matrix(1, 1)
Matrix sum(const Matrix& x);
void   sum(const Matrix& x, Matrix& out);

// Row max: Matrix(rows, cols) → Matrix(rows, 1)
Matrix row_max(const Matrix& x);
void   row_max(const Matrix& x, Matrix& out);

// Column max: Matrix(rows, cols) → Matrix(1, cols)
Matrix col_max(const Matrix& x);
void   col_max(const Matrix& x, Matrix& out);

// Total max: chains row_max then col_max → Matrix(1, 1)
Matrix max(const Matrix& x);
void   max(const Matrix& x, Matrix& out);
