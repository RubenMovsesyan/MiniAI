#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <matrix/matrix.cuh>

#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

static const char* DATA_ROOT = "src/matrix/tests/data";

// ─── CSV utilities ────────────────────────────────────────────────────────────

f32* csvLoad(const char* path, i32* out_rows, i32* out_cols) {
    FILE* f = fopen(path, "r");
    if (!f) { RLOG(LL_ERROR, "csvLoad: cannot open %s", path); return nullptr; }

    char* line = (char*)malloc(1 << 20);
    i32 rows = 0, cols = 0;
    while (fgets(line, 1 << 20, f)) {
        if (rows == 0) {
            cols = 1;
            for (char* p = line; *p; p++) if (*p == ',') cols++;
        }
        rows++;
    }
    rewind(f);

    f32* data = (f32*)malloc((usize)rows * cols * sizeof(f32));
    i32 idx = 0;
    while (fgets(line, 1 << 20, f)) {
        char* tok = strtok(line, ",\n\r");
        while (tok) { data[idx++] = (f32)atof(tok); tok = strtok(nullptr, ",\n\r"); }
    }
    fclose(f);
    free(line);
    *out_rows = rows;
    *out_cols = cols;
    return data;
}

static Matrix matLoad(const char* path) {
    i32 rows, cols;
    f32* h = csvLoad(path, &rows, &cols);
    Matrix m(rows, cols);
    cudaMemcpy(m.data, h, (usize)rows * cols * sizeof(f32), cudaMemcpyHostToDevice);
    free(h);
    return m;
}

bool matCheckCSV(const Matrix& result, const char* expected_path, f32 epsilon = 1e-4f) {
    i32 exp_rows, exp_cols;
    f32* h_exp = csvLoad(expected_path, &exp_rows, &exp_cols);
    if (!h_exp) return false;
    if (exp_rows != result.rows() || exp_cols != result.cols()) {
        RLOG(LL_ERROR, "shape mismatch: result %dx%d vs expected %dx%d",
             result.rows(), result.cols(), exp_rows, exp_cols);
        free(h_exp); return false;
    }
    i32 n = exp_rows * exp_cols;
    f32* h_res = (f32*)malloc((usize)n * sizeof(f32));
    cudaMemcpy(h_res, result.data, (usize)n * sizeof(f32), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (i32 i = 0; i < n && ok; i++) {
        if (fabsf(h_res[i] - h_exp[i]) > epsilon) {
            RLOG(LL_ERROR, "mismatch at element %d: got %.6f expected %.6f",
                 i, (double)h_res[i], (double)h_exp[i]);
            ok = false;
        }
    }
    free(h_res); free(h_exp);
    return ok;
}

// ─── Test harness ─────────────────────────────────────────────────────────────

static i32 s_pass = 0;
static i32 s_fail = 0;

static void record(bool ok, const char* label) {
    if (ok) { RLOG(LL_INFO,  "[ PASS ] %s", label); s_pass++; }
    else    { RLOG(LL_ERROR, "[ FAIL ] %s", label); s_fail++; }
}

// ─── Per-operation tests ──────────────────────────────────────────────────────

static void test_add(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    snprintf(pA,    sizeof(pA),    "%s/inputs/%s/A.csv",         DATA_ROOT, name);
    snprintf(pB,    sizeof(pB),    "%s/inputs/%s/B.csv",         DATA_ROOT, name);
    snprintf(pE,    sizeof(pE),    "%s/expected/add/%s.csv",     DATA_ROOT, name);
    snprintf(label, sizeof(label), "add/%s", name);
    Matrix A = matLoad(pA), B = matLoad(pB), C(A.rows(), A.cols());
    C = A + B;
    record(matCheckCSV(C, pE), label);
}

static void test_sub(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    snprintf(pA,    sizeof(pA),    "%s/inputs/%s/A.csv",         DATA_ROOT, name);
    snprintf(pB,    sizeof(pB),    "%s/inputs/%s/B.csv",         DATA_ROOT, name);
    snprintf(pE,    sizeof(pE),    "%s/expected/sub/%s.csv",     DATA_ROOT, name);
    snprintf(label, sizeof(label), "sub/%s", name);
    Matrix A = matLoad(pA), B = matLoad(pB), C(A.rows(), A.cols());
    C = A - B;
    record(matCheckCSV(C, pE), label);
}

static void test_hadamard(const char* name) {
    char pA[512], pB[512], pE[512], label[512];
    snprintf(pA,    sizeof(pA),    "%s/inputs/%s/A.csv",             DATA_ROOT, name);
    snprintf(pB,    sizeof(pB),    "%s/inputs/%s/B.csv",             DATA_ROOT, name);
    snprintf(pE,    sizeof(pE),    "%s/expected/hadamard/%s.csv",    DATA_ROOT, name);
    snprintf(label, sizeof(label), "hadamard/%s", name);
    Matrix A = matLoad(pA), B = matLoad(pB), C(A.rows(), A.cols());
    C = A.hadamard(B);
    record(matCheckCSV(C, pE), label);
}

static void test_transpose(const char* name) {
    char pA[512], pE[512], label[512];
    snprintf(pA,    sizeof(pA),    "%s/inputs/%s/A.csv",             DATA_ROOT, name);
    snprintf(pE,    sizeof(pE),    "%s/expected/transpose/%s.csv",   DATA_ROOT, name);
    snprintf(label, sizeof(label), "transpose/%s", name);
    Matrix A = matLoad(pA), C(A.cols(), A.rows());
    C = A.transpose();
    record(matCheckCSV(C, pE), label);
}

static void test_scalar_mul(const char* name) {
    char pA[512], pE[512], label[512];
    snprintf(pA, sizeof(pA), "%s/inputs/%s/A.csv", DATA_ROOT, name);
    Matrix A = matLoad(pA), C(A.rows(), A.cols());

    struct { f32 val; const char* op; } scalars[] = {
        { 2.0f,  "scalar_x2.0"  },
        { -1.5f, "scalar_x-1.5" },
        { 0.5f,  "scalar_x0.5"  },
    };
    for (auto& s : scalars) {
        snprintf(pE,    sizeof(pE),    "%s/expected/%s/%s.csv", DATA_ROOT, s.op, name);
        snprintf(label, sizeof(label), "%s/%s", s.op, name);
        C = A * s.val;
        record(matCheckCSV(C, pE), label);
    }
}

static void test_colAdd(const char* name) {
    char pA[512], pCol[512], pE[512], label[512];
    snprintf(pA,    sizeof(pA),    "%s/inputs/%s/A.csv",           DATA_ROOT, name);
    snprintf(pCol,  sizeof(pCol),  "%s/inputs/%s/col.csv",         DATA_ROOT, name);
    snprintf(pE,    sizeof(pE),    "%s/expected/colAdd/%s.csv",    DATA_ROOT, name);
    snprintf(label, sizeof(label), "colAdd/%s", name);
    Matrix A = matLoad(pA), col = matLoad(pCol), C(A.rows(), A.cols());
    C = A.colAdd(col);
    record(matCheckCSV(C, pE), label);
}

static void test_rowAdd(const char* name) {
    char pA[512], pRow[512], pE[512], label[512];
    snprintf(pA,    sizeof(pA),    "%s/inputs/%s/A.csv",           DATA_ROOT, name);
    snprintf(pRow,  sizeof(pRow),  "%s/inputs/%s/row.csv",         DATA_ROOT, name);
    snprintf(pE,    sizeof(pE),    "%s/expected/rowAdd/%s.csv",    DATA_ROOT, name);
    snprintf(label, sizeof(label), "rowAdd/%s", name);
    Matrix A = matLoad(pA), row = matLoad(pRow), C(A.rows(), A.cols());
    C = A.rowAdd(row);
    record(matCheckCSV(C, pE), label);
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

    static_assert(std::is_same_v<decltype(std::declval<Matrix>() * std::declval<Matrix>()), MulExpr>);
    static_assert(std::is_same_v<decltype(std::declval<MulExpr>() * std::declval<Matrix>()), ChainedMul>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() + std::declval<Matrix>()), AddExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() - std::declval<Matrix>()), SubExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().hadamard(std::declval<Matrix>())), HadExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().transpose()), TrExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>() * 2.0f), ScaleExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().colAdd(std::declval<Matrix>())), ColAddExpr>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix>().rowAdd(std::declval<Matrix>())), RowAddExpr>);

    record(true, "expression_types_compile");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

static const char* TEST_VARIANTS[] = {
    // small — all 4 types
    "3x3_rand_f32",   "3x3_struct_f32",   "3x3_rand_i32",   "3x3_struct_i32",
    "3x5_rand_f32",   "3x5_struct_f32",   "3x5_rand_i32",   "3x5_struct_i32",
    "4x4_rand_f32",   "4x4_struct_f32",   "4x4_rand_i32",   "4x4_struct_i32",
    "4x7_rand_f32",   "4x7_struct_f32",   "4x7_rand_i32",   "4x7_struct_i32",
    "8x8_rand_f32",   "8x8_struct_f32",   "8x8_rand_i32",   "8x8_struct_i32",
    // medium — all 4 types
    "16x16_rand_f32",  "16x16_struct_f32",  "16x16_rand_i32",  "16x16_struct_i32",
    "16x32_rand_f32",  "16x32_struct_f32",  "16x32_rand_i32",  "16x32_struct_i32",
    "64x64_rand_f32",  "64x64_struct_f32",  "64x64_rand_i32",  "64x64_struct_i32",
    "128x128_rand_f32","128x128_struct_f32","128x128_rand_i32","128x128_struct_i32",
    "128x256_rand_f32","128x256_struct_f32","128x256_rand_i32","128x256_struct_i32",
    // large — rand_f32 only
    "512x512_rand_f32",
    "1024x1024_rand_f32",
    nullptr,
};

int main() {
    initLog();
    test_expression_types_compile();

    for (i32 i = 0; TEST_VARIANTS[i]; i++) {
        const char* name = TEST_VARIANTS[i];
        test_add(name);
        test_sub(name);
        test_hadamard(name);
        test_transpose(name);
        test_scalar_mul(name);
        test_colAdd(name);
        test_rowAdd(name);
    }

    RLOG(LL_INFO, "%d passed, %d failed", s_pass, s_fail);
    return s_fail > 0 ? 1 : 0;
}
