#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <nn/nn.cuh>
#include <harness_test.h>
#include <harness_csv.h>

#include <cmath>
#include <cstdio>

// ─── CSV harness (local copies; NN_DATA_ROOT avoids DATA_ROOT macro collision) ──

inline const char* NN_DATA_ROOT = "src/nn/tests/data";

#define NN_PA(buf, name)     snprintf(buf, sizeof(buf), "%s/inputs/%s/A.csv",   NN_DATA_ROOT, name)
#define NN_PE(buf, op, name) snprintf(buf, sizeof(buf), "%s/expected/%s/%s.csv", NN_DATA_ROOT, op, name)
#define NN_LBL(buf, op, name) snprintf(buf, sizeof(buf), "%s/%s", op, name)

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

// Verify every output row sums to 1.0 within tolerance (and is finite).
inline bool checkRowSums(const Matrix& result, f32 epsilon = 1e-4f) {
    i32 rows = result.rows(), cols = result.cols();
    i32 n = rows * cols;
    f32* h = (f32*)malloc((usize)n * sizeof(f32));
    cudaMemcpy(h, result.data, (usize)n * sizeof(f32), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (i32 r = 0; r < rows && ok; r++) {
        f32 sum = 0.0f;
        for (i32 c = 0; c < cols; c++) {
            f32 v = h[r * cols + c];
            if (!isfinite(v)) { RLOG(LL_ERROR, "non-finite at row %d col %d: %f", r, c, v); ok = false; break; }
            sum += v;
        }
        if (ok && fabsf(sum - 1.0f) > epsilon) {
            RLOG(LL_ERROR, "row %d sums to %f, expected 1.0", r, (double)sum);
            ok = false;
        }
    }
    free(h);
    return ok;
}

// ─── Eager relu tests ────────────────────────────────────────────────────────

static void test_relu_eager() {
    const i32 rows = 4, cols = 3;
    f32 x_data[12] = {-2.0f, -1.0f, 0.0f, 1.0f, 0.5f, 2.0f, -0.5f, 3.0f, 0.0f, 1.5f, -3.0f, 2.5f};
    f32 expected[12] = {0.0f, 0.0f, 0.0f, 1.0f, 0.5f, 2.0f, 0.0f, 3.0f, 0.0f, 1.5f, 0.0f, 2.5f};

    Matrix x(rows, cols), y(rows, cols);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    // Eager path
    y = relu(x);

    f32 y_host[12];
    cudaMemcpy(y_host, y.data, sizeof(y_host), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (i32 i = 0; i < 12; i++) {
        if (fabs(y_host[i] - expected[i]) > 1e-5f) {
            RLOG(LL_ERROR, "relu[%d] = %f, expected %f", i, y_host[i], expected[i]);
            ok = false;
        }
    }
    record(ok, "eager_relu");
}

// ─── Eager relu tests (second variant, different range) ───────────────────────

static void test_relu_eager_v2() {
    const i32 rows = 3, cols = 4;
    f32 x_data[12] = {-5.0f, -2.5f, 0.0f, 1.0f, 2.0f, 3.5f, -1.0f, 0.0f, 10.0f, -0.5f, 5.0f, -3.0f};
    f32 expected[12] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.5f, 0.0f, 0.0f, 10.0f, 0.0f, 5.0f, 0.0f};

    Matrix x(rows, cols), y(rows, cols);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    y = relu(x);

    f32 y_host[12];
    cudaMemcpy(y_host, y.data, sizeof(y_host), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (i32 i = 0; i < 12; i++) {
        if (fabs(y_host[i] - expected[i]) > 1e-5f) {
            RLOG(LL_ERROR, "relu_v2[%d] = %f, expected %f", i, y_host[i], expected[i]);
            ok = false;
        }
    }
    record(ok, "relu_eager_v2");
}

// ─── Grad relu tests ────────────────────────────────────────────────────────

static void test_grad_relu() {
    const i32 rows = 4, cols = 3;
    f32 x_data[12] = {-2.0f, -1.0f, 0.0f, 1.0f, 0.5f, 2.0f, -0.5f, 3.0f, 0.0f, 1.5f, -3.0f, 2.5f};
    f32 dy_data[12] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    f32 expected[12] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};

    Matrix x(rows, cols), dy(rows, cols), dx(rows, cols);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);
    cudaMemcpy(dy.data, dy_data, sizeof(dy_data), cudaMemcpyHostToDevice);

    dx = grad_relu(x, dy);

    f32 dx_host[12];
    cudaMemcpy(dx_host, dx.data, sizeof(dx_host), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (i32 i = 0; i < 12; i++) {
        if (fabs(dx_host[i] - expected[i]) > 1e-5f) {
            RLOG(LL_ERROR, "grad_relu[%d] = %f, expected %f", i, dx_host[i], expected[i]);
            ok = false;
        }
    }
    record(ok, "grad_relu");
}

// ─── Out-param variants ──────────────────────────────────────────────────────

static void test_relu_outparam() {
    const i32 rows = 2, cols = 2;
    f32 x_data[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    f32 expected[4] = {0.0f, 0.0f, 1.0f, 2.0f};

    Matrix x(rows, cols), y(rows, cols);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    relu(x, y);

    f32 y_host[4];
    cudaMemcpy(y_host, y.data, sizeof(y_host), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (i32 i = 0; i < 4; i++) {
        if (fabs(y_host[i] - expected[i]) > 1e-5f) ok = false;
    }
    record(ok, "relu_outparam");
}

// ─── Softmax CSV tests ─────────────────────────────────────────────────────────

static void test_softmax(const char* name) {
    char pA[512], pE[512], label[512];
    NN_PA(pA, name); NN_PE(pE, "softmax", name); NN_LBL(label, "softmax", name);
    Matrix A = matLoad(pA), C(A.rows(), A.cols());
    C = softmax(A);
    bool ok = matCheckCSV(C, pE, 1e-5f) && checkRowSums(C);
    record(ok, label);
}

static void test_softmax_lazy(const char* name) {
    char pA[512], pE[512], label[512];
    NN_PA(pA, name); NN_PE(pE, "softmax", name); NN_LBL(label, "softmax_lazy", name);
    Matrix A = matLoad(pA);
    Matrix C = SoftmaxExpr<MatrixRef>(A.ref()).eval();   // exercises materialize path
    bool ok = matCheckCSV(C, pE, 1e-5f) && checkRowSums(C);
    record(ok, label);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);
    test_relu_eager();
    test_relu_eager_v2();
    test_grad_relu();
    test_relu_outparam();

    const char* softmax_variants[] = {
        "2x3_f32", "3x4_f32", "10x10_f32", "100x100_f32",
        "1x100_f32", "100x1_f32", "8x8_large_f32", nullptr,
    };
    for (const char** v = softmax_variants; *v; v++)
        test_softmax(*v);
    test_softmax_lazy("3x4_f32");

    return testSummary();
}
