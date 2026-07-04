#pragma once

#include <common.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>
#include <vector>

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

    // Named lazy adapters (aliases of the operators) so a .lazy() chain reads
    // A.lazy().mul(B).add(C) ; and eval() forces the tree into an owned Matrix.
    template <typename RHS>
    MatrixMulExpr<Derived, NodeOf_t<RHS>>      mul(const RHS& rhs)       const;  // matmul
    MatrixScalarMulExpr<Derived>               scale(f32 s)             const;
    template <typename RHS>
    MatrixAddExpr<Derived, NodeOf_t<RHS>>      add(const RHS& rhs)       const;
    template <typename RHS>
    MatrixSubExpr<Derived, NodeOf_t<RHS>>      sub(const RHS& rhs)       const;
    Matrix                                     eval()                    const;
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
    MatrixRef lazy()      const { return ref(); }  // gateway into the lazy/fused expression world
    operator  MatrixRef() const { return ref(); }

    // Lazy operators — build expression nodes (fused at operator=). Go through ref()
    // so trees use MatrixRef leaves. Bodies defined out-of-class below.
    template <typename RHS> auto operator*(const RHS& rhs) const;
    auto                         operator*(f32 s)          const;
    template <typename RHS> auto operator+(const RHS& rhs) const;
    template <typename RHS> auto operator-(const RHS& rhs) const;

    // Eager runtime ops — run now on the GPU, return an owned Matrix. Each has an
    // out-param overload writing into a preallocated Matrix (no allocation). Bodies
    // reuse operator= so each still runs as one fused kernel internally.
    Matrix matmul  (const Matrix& b) const;  void matmul  (const Matrix& b, Matrix& out) const;
    Matrix add     (const Matrix& b) const;  void add     (const Matrix& b, Matrix& out) const;
    Matrix sub     (const Matrix& b) const;  void sub     (const Matrix& b, Matrix& out) const;
    Matrix hadamard(const Matrix& b) const;  void hadamard(const Matrix& b, Matrix& out) const;
    Matrix scale   (f32 s)           const;  void scale   (f32 s,           Matrix& out) const;
    Matrix transposed()              const;  void transposed(              Matrix& out) const;
    Matrix colAdd  (const Matrix& c) const;  void colAdd  (const Matrix& c, Matrix& out) const;
    Matrix rowAdd  (const Matrix& r) const;  void rowAdd  (const Matrix& r, Matrix& out) const;

    // Evaluate any expression into this matrix's existing allocation via a fused kernel.
    // Precondition: expr.rows() == rows() && expr.cols() == cols()
    template <typename LHS, typename RHS>
    Matrix& operator=(const MatrixMulExpr<LHS, RHS>& expr);

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

// Named lazy adapters — thin aliases of the operators.
template <typename Derived> template <typename RHS>
MatrixMulExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::mul(const RHS& rhs) const { return {self(), nodeOf(rhs)}; }

template <typename Derived>
MatrixScalarMulExpr<Derived>
MatrixExpr<Derived>::scale(f32 s) const { return {self(), s}; }

template <typename Derived> template <typename RHS>
MatrixAddExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::add(const RHS& rhs) const { return {self(), nodeOf(rhs)}; }

template <typename Derived> template <typename RHS>
MatrixSubExpr<Derived, NodeOf_t<RHS>>
MatrixExpr<Derived>::sub(const RHS& rhs) const { return {self(), nodeOf(rhs)}; }

// ─── Matrix forwarding operator definitions ───────────────────────────────────

template <typename RHS>
auto Matrix::operator*(const RHS& rhs) const { return ref() * rhs; }
inline auto Matrix::operator*(f32 s) const    { return ref() * s; }
template <typename RHS>
auto Matrix::operator+(const RHS& rhs) const { return ref() + rhs; }
template <typename RHS>
auto Matrix::operator-(const RHS& rhs) const { return ref() - rhs; }

// ─── GEMM dispatch ────────────────────────────────────────────────────────────
void matmulDispatch(const f32* A, const f32* B, f32* C, i32 M, i32 K, i32 N);

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

// ─── Matmul materialization ───────────────────────────────────────────────────
// Matmul can't fuse element-wise (needs a K-reduction). Before launching the fused
// element-wise kernel, walk the expression tree and pre-evaluate every MatrixMulExpr
// into its own GPU buffer, replacing that node with a MatrixRef. Element-wise nodes
// keep their structure so they still fuse into one matEvalKernel.
//
//   materialize(node)      → equivalent tree with matmul nodes replaced by MatrixRef
//   materializeToRef(node) → a single MatrixRef to a fully-evaluated buffer
//
// `temps` owns every intermediate buffer; it must outlive the kernel that reads them.
// Matrix move preserves its data pointer, so vector reallocation never invalidates a ref.

template <typename T>             struct is_matmul_expr                        : std::false_type {};
template <typename L, typename R> struct is_matmul_expr<MatrixMulExpr<L, R>>   : std::true_type  {};

// Forward decls (materialize and materializeToRef are mutually recursive).
inline MatrixRef materialize(const MatrixRef& n, std::vector<Matrix>&);
template <typename L, typename R> MatrixRef materialize(const MatrixMulExpr<L, R>& n, std::vector<Matrix>& t);
template <typename L, typename R> auto materialize(const MatrixAddExpr<L, R>& n, std::vector<Matrix>& t);
template <typename L, typename R> auto materialize(const MatrixSubExpr<L, R>& n, std::vector<Matrix>& t);
template <typename L, typename R> auto materialize(const MatrixHadamardExpr<L, R>& n, std::vector<Matrix>& t);
template <typename L, typename R> auto materialize(const MatrixColAddExpr<L, R>& n, std::vector<Matrix>& t);
template <typename L, typename R> auto materialize(const MatrixRowAddExpr<L, R>& n, std::vector<Matrix>& t);
template <typename L>             auto materialize(const MatrixScalarMulExpr<L>& n, std::vector<Matrix>& t);
template <typename L>             auto materialize(const MatrixTransposeExpr<L>& n, std::vector<Matrix>& t);

inline MatrixRef materializeToRef(const MatrixRef& n, std::vector<Matrix>&);
template <typename L, typename R> MatrixRef materializeToRef(const MatrixMulExpr<L, R>& n, std::vector<Matrix>& t);
template <typename E>
    requires (!is_matmul_expr<E>::value && !std::is_same_v<E, MatrixRef>)
MatrixRef materializeToRef(const E& n, std::vector<Matrix>& t);

// ── materialize: leaf and matmul ──
inline MatrixRef materialize(const MatrixRef& n, std::vector<Matrix>&) { return n; }

template <typename L, typename R>
MatrixRef materialize(const MatrixMulExpr<L, R>& n, std::vector<Matrix>& t) {
    return materializeToRef(n, t);
}

// ── materialize: element-wise nodes rebuilt with materialized children ──
// Materialize each child ONCE into a local (calling materialize twice would double-GEMM),
// then reconstruct the node with the (possibly rewritten) child types.
template <typename L, typename R>
auto materialize(const MatrixAddExpr<L, R>& n, std::vector<Matrix>& t) {
    auto l = materialize(n.lhs, t); auto r = materialize(n.rhs, t);
    return MatrixAddExpr<decltype(l), decltype(r)>(l, r);
}
template <typename L, typename R>
auto materialize(const MatrixSubExpr<L, R>& n, std::vector<Matrix>& t) {
    auto l = materialize(n.lhs, t); auto r = materialize(n.rhs, t);
    return MatrixSubExpr<decltype(l), decltype(r)>(l, r);
}
template <typename L, typename R>
auto materialize(const MatrixHadamardExpr<L, R>& n, std::vector<Matrix>& t) {
    auto l = materialize(n.lhs, t); auto r = materialize(n.rhs, t);
    return MatrixHadamardExpr<decltype(l), decltype(r)>(l, r);
}
template <typename L, typename R>
auto materialize(const MatrixColAddExpr<L, R>& n, std::vector<Matrix>& t) {
    auto l = materialize(n.lhs, t); auto c = materialize(n.col, t);
    return MatrixColAddExpr<decltype(l), decltype(c)>(l, c);
}
template <typename L, typename R>
auto materialize(const MatrixRowAddExpr<L, R>& n, std::vector<Matrix>& t) {
    auto l = materialize(n.lhs, t); auto r = materialize(n.row, t);
    return MatrixRowAddExpr<decltype(l), decltype(r)>(l, r);
}
template <typename L>
auto materialize(const MatrixScalarMulExpr<L>& n, std::vector<Matrix>& t) {
    auto l = materialize(n.lhs, t);
    return MatrixScalarMulExpr<decltype(l)>(l, n.scalar);
}
template <typename L>
auto materialize(const MatrixTransposeExpr<L>& n, std::vector<Matrix>& t) {
    auto l = materialize(n.lhs, t);
    return MatrixTransposeExpr<decltype(l)>(l);
}

// ── materializeToRef: collapse any subtree to a single evaluated buffer ──
inline MatrixRef materializeToRef(const MatrixRef& n, std::vector<Matrix>&) { return n; }  // no copy

template <typename L, typename R>
MatrixRef materializeToRef(const MatrixMulExpr<L, R>& n, std::vector<Matrix>& t) {
    MatrixRef a = materializeToRef(n.lhs, t);
    MatrixRef b = materializeToRef(n.rhs, t);
    i32 M = n.lhs.rows(), K = n.lhs.cols(), N = n.rhs.cols();
    Matrix tmp(M, N);
    matmulDispatch(a.data, b.data, tmp.data, M, K, N);
    t.push_back(std::move(tmp));
    return t.back().ref();
}

template <typename E>
    requires (!is_matmul_expr<E>::value && !std::is_same_v<E, MatrixRef>)
MatrixRef materializeToRef(const E& n, std::vector<Matrix>& t) {
    auto m = materialize(n, t);  // strip any inner matmuls first
    Matrix tmp(n.rows(), n.cols());
    dim3 block(16, 16);
    dim3 grid((n.cols() + 15) / 16, (n.rows() + 15) / 16);
    matEvalKernel<<<grid, block>>>(m, tmp.data, tmp.rows(), tmp.cols());
    cudaDeviceSynchronize();
    t.push_back(std::move(tmp));
    return t.back().ref();
}

// ─── Matrix::operator= ────────────────────────────────────────────────────────

// Matmul top node → GEMM straight into this->data; operands collapsed to refs.
template <typename LHS, typename RHS>
Matrix& Matrix::operator=(const MatrixMulExpr<LHS, RHS>& expr) {
    std::vector<Matrix> temps;
    MatrixRef a = materializeToRef(expr.lhs, temps);
    MatrixRef b = materializeToRef(expr.rhs, temps);
    matmulDispatch(a.data, b.data, data, _rows, expr.lhs.cols(), _cols);
    return *this;  // matmulDispatch synced before temps die
}

// Element-wise top node → rewrite away inner matmuls, then one fused kernel.
template <typename Expr>
    requires (!std::is_same_v<std::decay_t<Expr>, Matrix>)
Matrix& Matrix::operator=(const Expr& expr) {
    std::vector<Matrix> temps;
    auto m = materialize(expr, temps);
    dim3 block(16, 16);
    dim3 grid((_cols + 15) / 16, (_rows + 15) / 16);
    matEvalKernel<<<grid, block>>>(m, data, _rows, _cols);
    cudaDeviceSynchronize();
    return *this;
}

// ─── MatrixExpr::eval ─────────────────────────────────────────────────────────
// Force a lazy expression into a fresh owned Matrix (the .lazy() chain terminal).
// (Distinct from Matrix::eval(), which clones an existing Matrix device→device.)
template <typename Derived>
Matrix MatrixExpr<Derived>::eval() const {
    Matrix out(self().rows(), self().cols());
    out = self();  // routes through the matmul / element-wise operator= overloads
    return out;
}

// ─── Eager runtime ops ────────────────────────────────────────────────────────
// Each builds the matching lazy node and forces it via operator= (one fused kernel).
// Owned-return variants allocate the result; out-param variants reuse a preallocated
// Matrix (assumed correct shape — same contract as matmulDispatch).

inline Matrix Matrix::matmul(const Matrix& b) const {
    Matrix out(_rows, b._cols); out = ref() * b.ref(); return out;
}
inline void Matrix::matmul(const Matrix& b, Matrix& out) const { out = ref() * b.ref(); }

inline Matrix Matrix::add(const Matrix& b) const {
    Matrix out(_rows, _cols); out = ref() + b.ref(); return out;
}
inline void Matrix::add(const Matrix& b, Matrix& out) const { out = ref() + b.ref(); }

inline Matrix Matrix::sub(const Matrix& b) const {
    Matrix out(_rows, _cols); out = ref() - b.ref(); return out;
}
inline void Matrix::sub(const Matrix& b, Matrix& out) const { out = ref() - b.ref(); }

inline Matrix Matrix::hadamard(const Matrix& b) const {
    Matrix out(_rows, _cols); out = ref().hadamard(b.ref()); return out;
}
inline void Matrix::hadamard(const Matrix& b, Matrix& out) const { out = ref().hadamard(b.ref()); }

inline Matrix Matrix::scale(f32 s) const {
    Matrix out(_rows, _cols); out = ref() * s; return out;
}
inline void Matrix::scale(f32 s, Matrix& out) const { out = ref() * s; }

inline Matrix Matrix::transposed() const {
    Matrix out(_cols, _rows); out = ref().transpose(); return out;
}
inline void Matrix::transposed(Matrix& out) const { out = ref().transpose(); }

inline Matrix Matrix::colAdd(const Matrix& c) const {
    Matrix out(_rows, _cols); out = ref().colAdd(c.ref()); return out;
}
inline void Matrix::colAdd(const Matrix& c, Matrix& out) const { out = ref().colAdd(c.ref()); }

inline Matrix Matrix::rowAdd(const Matrix& r) const {
    Matrix out(_rows, _cols); out = ref().rowAdd(r.ref()); return out;
}
inline void Matrix::rowAdd(const Matrix& r, Matrix& out) const { out = ref().rowAdd(r.ref()); }
