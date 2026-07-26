#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <agg/agg.cuh>
#include <harness_test.h>
#include <harness_csv.h>

#include <cstdio>

inline const char* AGG_DATA_ROOT = "src/agg/tests/data";

#define PA(buf, name)     snprintf(buf, sizeof(buf), "%s/inputs/%s/A.csv",   AGG_DATA_ROOT, name)
#define PE(buf, op, name) snprintf(buf, sizeof(buf), "%s/expected/%s/%s.csv", AGG_DATA_ROOT, op, name)
#define LBL(buf, op, name) snprintf(buf, sizeof(buf), "%s/%s", op, name)

inline Matrix matLoad(const char* path) {
    i32 rows, cols;
    f32* h = csvLoad(path, &rows, &cols);
    Matrix m(rows, cols);
    cudaMemcpy(m.data, h, (usize)rows * cols * sizeof(f32), cudaMemcpyHostToDevice);
    free(h);
    return m;
}

inline bool matCheckCSV(const Matrix& result, const char* expected_path, f32 epsilon = 1e-4f) {
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
        if (fabsf(h_res[i] - h_exp[i]) > epsilon * fmaxf(1.0f, fabsf(h_exp[i]))) {
            RLOG(LL_ERROR, "mismatch at element %d: got %.6f expected %.6f",
                 i, (double)h_res[i], (double)h_exp[i]);
            ok = false;
        }
    }
    free(h_res); free(h_exp);
    return ok;
}

// ─── Row sum CSV tests ────────────────────────────────────────────────────

static void test_row_sum(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "row_sum", name); LBL(label, "row_sum", name);
    Matrix A = matLoad(pA), C(A.rows(), 1);
    C = row_sum(A);
    record(matCheckCSV(C, pE), label);
}

// ─── Column sum CSV tests ──────────────────────────────────────────────────

static void test_col_sum(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "col_sum", name); LBL(label, "col_sum", name);
    Matrix A = matLoad(pA), C(1, A.cols());
    C = col_sum(A);
    record(matCheckCSV(C, pE), label);
}

// ─── Total sum CSV tests ──────────────────────────────────────────────────

static void test_sum(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "sum", name); LBL(label, "sum", name);
    Matrix A = matLoad(pA), C(1, 1);
    C = sum(A);
    record(matCheckCSV(C, pE), label);
}

// ─── Row max CSV tests ────────────────────────────────────────────────────

static void test_row_max(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "row_max", name); LBL(label, "row_max", name);
    Matrix A = matLoad(pA), C(A.rows(), 1);
    C = row_max(A);
    record(matCheckCSV(C, pE), label);
}

// ─── Column max CSV tests ──────────────────────────────────────────────────

static void test_col_max(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "col_max", name); LBL(label, "col_max", name);
    Matrix A = matLoad(pA), C(1, A.cols());
    C = col_max(A);
    record(matCheckCSV(C, pE), label);
}

// ─── Total max CSV tests ───────────────────────────────────────────────────

static void test_max(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "max", name); LBL(label, "max", name);
    Matrix A = matLoad(pA), C(1, 1);
    C = max(A);
    record(matCheckCSV(C, pE), label);
}

// ─── Row argmax CSV tests ──────────────────────────────────────────────────

static void test_row_argmax(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "row_argmax", name); LBL(label, "row_argmax", name);
    Matrix A = matLoad(pA), C(A.rows(), 1);
    C = row_argmax(A);
    record(matCheckCSV(C, pE), label);
}

// ─── Out-param variants ───────────────────────────────────────────────────

static void test_row_sum_outparam(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "row_sum", name); LBL(label, "row_sum_outparam", name);
    Matrix A = matLoad(pA), C(A.rows(), 1);
    row_sum(A, C);
    record(matCheckCSV(C, pE), label);
}

static void test_row_max_outparam(const char* name) {
    char pA[512], pE[512], label[512];
    PA(pA, name); PE(pE, "row_max", name); LBL(label, "row_max_outparam", name);
    Matrix A = matLoad(pA), C(A.rows(), 1);
    row_max(A, C);
    record(matCheckCSV(C, pE), label);
}

// ─── Variant list ─────────────────────────────────────────────────────────

inline const char* AGG_VARIANTS[] = {
    "2x3_f32",
    "3x4_f32",
    "10x10_f32",
    "100x100_f32",
    "1x100_f32",
    "100x1_f32",
    nullptr,
};

// ─── Main ─────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);

    for (const char** var = AGG_VARIANTS; *var; var++) {
        test_row_sum(*var);
        test_col_sum(*var);
        test_sum(*var);
        test_row_max(*var);
        test_col_max(*var);
        test_max(*var);
        test_row_argmax(*var);
        test_row_sum_outparam(*var);
        test_row_max_outparam(*var);
    }

    return testSummary();
}
