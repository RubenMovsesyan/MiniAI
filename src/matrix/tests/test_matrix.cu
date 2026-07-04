#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <matrix/matrix.cuh>
#include <harness_test.h>
#include <harness_matrix_csv.cuh>

#include <type_traits>
#include <cstdio>

// ─── Single-operation tests ───────────────────────────────────────────────────

static void test_add(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"add",name); LBL(label,"add",name);
    Matrix A = matLoad(pA), B = matLoad(pB), C(A.rows(), A.cols());
    C = A + B;
    record(matCheckCSV(C, pE), label);
}

static void test_sub(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"sub",name); LBL(label,"sub",name);
    Matrix A = matLoad(pA), B = matLoad(pB), C(A.rows(), A.cols());
    C = A - B;
    record(matCheckCSV(C, pE), label);
}

static void test_hadamard(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"hadamard",name); LBL(label,"hadamard",name);
    Matrix A = matLoad(pA), B = matLoad(pB), C(A.rows(), A.cols());
    C = A.hadamard(B);
    record(matCheckCSV(C, pE), label);
}

static void test_transpose(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA,name); PE(pE,"transpose",name); LBL(label,"transpose",name);
    Matrix A = matLoad(pA);
    Matrix C = A.transposed();   // eager
    record(matCheckCSV(C, pE), label);
}

static void test_scalar_mul(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name);
    Matrix A = matLoad(pA), C(A.rows(), A.cols());

    struct { f32 val; const char* op; } scalars[] = {
        { 2.0f,  "scalar_x2.0"  },
        { -1.5f, "scalar_x-1.5" },
        { 0.5f,  "scalar_x0.5"  },
    };
    for (auto& s : scalars) {
        PE(pE, s.op, name); LBL(label, s.op, name);
        C = A * s.val;
        record(matCheckCSV(C, pE), label);
    }
}

static void test_colAdd(const char* name) {
    char pA[512], pCol[512], pE[512], label[512];
    PA(pA,name); PCOL(pCol,name); PE(pE,"colAdd",name); LBL(label,"colAdd",name);
    Matrix A = matLoad(pA), col = matLoad(pCol), C(A.rows(), A.cols());
    C = A.colAdd(col);
    record(matCheckCSV(C, pE), label);
}

static void test_rowAdd(const char* name) {
    char pA[512], pRow[512], pE[512], label[512];
    PA(pA,name); PROW(pRow,name); PE(pE,"rowAdd",name); LBL(label,"rowAdd",name);
    Matrix A = matLoad(pA), row = matLoad(pRow), C(A.rows(), A.cols());
    C = A.rowAdd(row);
    record(matCheckCSV(C, pE), label);
}

// ─── Matrix multiply (square only) ───────────────────────────────────────────
// Stub returns 0 — these tests FAIL until the real GEMM kernel is wired in.

static void test_matmul(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"matmul",name); LBL(label,"matmul",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = A * B;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE, 1e-2f), label);
}

// ─── Element-wise chain tests (all sizes/types) ───────────────────────────────
// These only use element-wise ops and PASS.

static void test_chain_add_scale(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_add_scale",name); LBL(label,"chain_add_scale",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = (A + B) * 2.0f;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE), label);
}

static void test_chain_sub_scale(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_sub_scale",name); LBL(label,"chain_sub_scale",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = (A - B) * 0.5f;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE), label);
}

static void test_chain_transpose_scale(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA,name); PE(pE,"chain_transpose_scale",name); LBL(label,"chain_transpose_scale",name);
    Matrix A = matLoad(pA);
    auto expr = A.lazy().transpose() * 2.0f;   // lazy/fused
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE), label);
}

static void test_chain_scale_colAdd(const char* name) {
    char pA[512], pCol[512], pE[512], label[512];
    PA(pA,name); PCOL(pCol,name); PE(pE,"chain_scale_colAdd",name); LBL(label,"chain_scale_colAdd",name);
    Matrix A = matLoad(pA), col = matLoad(pCol);
    auto expr = (A * -1.5f).colAdd(col);
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE), label);
}

static void test_chain_colAdd_rowAdd(const char* name) {
    char pA[512], pCol[512], pRow[512], pE[512], label[512];
    PA(pA,name); PCOL(pCol,name); PROW(pRow,name);
    PE(pE,"chain_colAdd_rowAdd",name); LBL(label,"chain_colAdd_rowAdd",name);
    Matrix A = matLoad(pA), col = matLoad(pCol), row = matLoad(pRow);
    auto expr = A.lazy().colAdd(col).rowAdd(row);   // lazy/fused
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE), label);
}

static void test_chain_add_scale_sub(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_add_scale_sub",name); LBL(label,"chain_add_scale_sub",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = (A + B) * 2.0f - A;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE), label);
}

static void test_chain_sub_hadamard_scale(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_sub_hadamard_scale",name); LBL(label,"chain_sub_hadamard_scale",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = (A - B).hadamard(A) * 0.5f;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE), label);
}

static void test_chain_4_add_scale_sub_had(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name);
    PE(pE,"chain_4_add_scale_sub_had",name); LBL(label,"chain_4_add_scale_sub_had",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = ((A + B) * 2.0f - A).hadamard(B);
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE), label);
}

// ─── Matmul chain tests (square only) ─────────────────────────────────────────
// All FAIL until the real GEMM kernel is wired in.

static void test_chain_matmul_add(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_matmul_add",name); LBL(label,"chain_matmul_add",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = A * B + A;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE, 1e-2f), label);
}

static void test_chain_matmul_scale(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_matmul_scale",name); LBL(label,"chain_matmul_scale",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = (A * B) * 2.0f;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE, 1e-2f), label);
}

static void test_chain_add_matmul(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_add_matmul",name); LBL(label,"chain_add_matmul",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = (A + B) * A;   // matrix multiply: (A+B) @ A
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE, 1e-2f), label);
}

static void test_chain_matmul_add_scale(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_matmul_add_scale",name); LBL(label,"chain_matmul_add_scale",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = (A * B + A) * 2.0f;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE, 1e-2f), label);
}

static void test_chain_matmul_sub_scale(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"chain_matmul_sub_scale",name); LBL(label,"chain_matmul_sub_scale",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = (A * B - B) * 0.5f;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE, 1e-2f), label);
}

static void test_chain_matmul_add_scale_sub(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name);
    PE(pE,"chain_matmul_add_scale_sub",name); LBL(label,"chain_matmul_add_scale_sub",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    auto expr = ((A * B + A) * 2.0f) - B;
    Matrix C(expr.rows(), expr.cols());
    C = expr;
    record(matCheckCSV(C, pE, 1e-2f), label);
}

// ─── Eager (runtime) API tests ────────────────────────────────────────────────
// Eager verbs run now and return an owned Matrix; verified against the same CSV
// expecteds the lazy tests use (both paths run the same kernels).

static void test_eager_elementwise(const char* name) {
    char pA[512], pB[512], pCol[512], pRow[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PCOL(pCol,name); PROW(pRow,name);
    Matrix A = matLoad(pA), B = matLoad(pB), col = matLoad(pCol), row = matLoad(pRow);

    { Matrix r = A.add(B);       PE(pE,"add",name);         LBL(label,"eager_add",name);        record(matCheckCSV(r, pE), label); }
    { Matrix r = A.sub(B);       PE(pE,"sub",name);         LBL(label,"eager_sub",name);        record(matCheckCSV(r, pE), label); }
    { Matrix r = A.hadamard(B);  PE(pE,"hadamard",name);    LBL(label,"eager_hadamard",name);   record(matCheckCSV(r, pE), label); }
    { Matrix r = A.scale(2.0f);  PE(pE,"scalar_x2.0",name); LBL(label,"eager_scale",name);      record(matCheckCSV(r, pE), label); }
    { Matrix r = A.transposed(); PE(pE,"transpose",name);   LBL(label,"eager_transposed",name); record(matCheckCSV(r, pE), label); }
    { Matrix r = A.colAdd(col);  PE(pE,"colAdd",name);      LBL(label,"eager_colAdd",name);     record(matCheckCSV(r, pE), label); }
    { Matrix r = A.rowAdd(row);  PE(pE,"rowAdd",name);      LBL(label,"eager_rowAdd",name);     record(matCheckCSV(r, pE), label); }
}

static void test_eager_matmul(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"matmul",name); LBL(label,"eager_matmul",name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    Matrix r = A.matmul(B);
    record(matCheckCSV(r, pE, 1e-2f), label);
}

static void test_eager_matmul_outparam(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name); PE(pE,"matmul",name); LBL(label,"eager_matmul_out",name);
    Matrix A = matLoad(pA), B = matLoad(pB), out(A.rows(), B.cols());
    A.matmul(B, out);   // out-param overload — no allocation
    record(matCheckCSV(out, pE, 1e-2f), label);
}

static void test_eager_chain(const char* name) {
    // Eager chaining through owned returns; reuses existing chain expecteds.
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    { Matrix r = A.matmul(B).add(A);      PE(pE,"chain_matmul_add",name);   LBL(label,"eager_chain_matmul_add",name);   record(matCheckCSV(r, pE, 1e-2f), label); }
    { Matrix r = A.matmul(B).scale(2.0f); PE(pE,"chain_matmul_scale",name); LBL(label,"eager_chain_matmul_scale",name); record(matCheckCSV(r, pE, 1e-2f), label); }
}

static void test_lazy_gateway(const char* name) {
    // .lazy() ... .eval() must equal the operator form / expected data.
    char pA[512], pB[512], pE[512], label[512];
    PA(pA,name); PB(pB,name);
    Matrix A = matLoad(pA), B = matLoad(pB);
    { Matrix r = A.lazy().mul(B).eval();        PE(pE,"matmul",name);           LBL(label,"lazy_mul",name);     record(matCheckCSV(r, pE, 1e-2f), label); }
    { Matrix r = A.lazy().mul(B).add(A).eval(); PE(pE,"chain_matmul_add",name);  LBL(label,"lazy_mul_add",name); record(matCheckCSV(r, pE, 1e-2f), label); }
}

// ─── Compile-time type check ──────────────────────────────────────────────────

static void test_expression_types_compile() {
    // Expression tree leaves are MatrixRef (trivially copyable view), not Matrix.
    // Matrix operators go through ref() so expression trees never contain non-copyable Matrix.
    using Ref        = MatrixRef;
    using MulExpr    = MatrixMulExpr<Ref, Ref>;
    using ChainedMul = MatrixMulExpr<MulExpr, Ref>;
    using AddExpr    = MatrixAddExpr<Ref, Ref>;
    using SubExpr    = MatrixSubExpr<Ref, Ref>;
    using HadExpr    = MatrixHadamardExpr<Ref, Ref>;
    using TrExpr     = MatrixTransposeExpr<Ref>;
    using ScaleExpr  = MatrixScalarMulExpr<Ref>;
    using ColAddExpr = MatrixColAddExpr<Ref, Ref>;
    using RowAddExpr = MatrixRowAddExpr<Ref, Ref>;

    // Operators stay lazy (return expression nodes).
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() * std::declval<Matrix>()), MulExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MulExpr>() * std::declval<Matrix>()), ChainedMul>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() + std::declval<Matrix>()), AddExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() - std::declval<Matrix>()), SubExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() * 2.0f), ScaleExpr>);

    // Named lazy adapters live behind .lazy() (the MatrixRef expression view).
    static_assert(std::is_same_v<decltype(std::declval<MatrixRef>().mul(std::declval<Matrix>())), MulExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MatrixRef>().add(std::declval<Matrix>())), AddExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MatrixRef>().sub(std::declval<Matrix>())), SubExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MatrixRef>().scale(2.0f)), ScaleExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MatrixRef>().hadamard(std::declval<Matrix>())), HadExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MatrixRef>().transpose()), TrExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MatrixRef>().colAdd(std::declval<Matrix>())), ColAddExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MatrixRef>().rowAdd(std::declval<Matrix>())), RowAddExpr>);

    // Plain-verb methods on Matrix are eager → return an owned Matrix.
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().matmul(std::declval<Matrix>())), Matrix>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().add(std::declval<Matrix>())), Matrix>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().hadamard(std::declval<Matrix>())), Matrix>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().transposed()), Matrix>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().scale(2.0f)), Matrix>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().colAdd(std::declval<Matrix>())), Matrix>);
    // .lazy() enters the expression world; .eval() forces it back to a Matrix.
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().lazy()), MatrixRef>);
    static_assert(std::is_same_v<decltype(std::declval<MulExpr>().eval()), Matrix>);

    record(true, "expression_types_compile");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog();
    test_expression_types_compile();

    // Single-operation tests
    for (i32 i = 0; ALL_VARIANTS[i]; i++) {
        const char* n = ALL_VARIANTS[i];
        test_add(n);
        test_sub(n);
        test_hadamard(n);
        test_transpose(n);
        test_scalar_mul(n);
        test_colAdd(n);
        test_rowAdd(n);
        test_eager_elementwise(n);
    }

    // Element-wise chain tests (all sizes — will PASS)
    for (i32 i = 0; ALL_VARIANTS[i]; i++) {
        const char* n = ALL_VARIANTS[i];
        test_chain_add_scale(n);
        test_chain_sub_scale(n);
        test_chain_transpose_scale(n);
        test_chain_scale_colAdd(n);
        test_chain_colAdd_rowAdd(n);
        test_chain_add_scale_sub(n);
        test_chain_sub_hadamard_scale(n);
        test_chain_4_add_scale_sub_had(n);
    }

    // Matmul and matmul-chain tests (square only — will FAIL until GEMM is implemented)
    for (i32 i = 0; SQUARE_VARIANTS[i]; i++) {
        const char* n = SQUARE_VARIANTS[i];
        test_matmul(n);
        test_chain_matmul_add(n);
        test_chain_matmul_scale(n);
        test_chain_add_matmul(n);
        test_chain_matmul_add_scale(n);
        test_chain_matmul_sub_scale(n);
        test_chain_matmul_add_scale_sub(n);
        test_eager_matmul(n);
        test_eager_matmul_outparam(n);
        test_eager_chain(n);
        test_lazy_gateway(n);
    }

    return testSummary();
}
