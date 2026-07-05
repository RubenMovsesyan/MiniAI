#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <agg/agg.cuh>
#include <harness_test.h>

#include <cmath>

// ─── Row sum tests ────────────────────────────────────────────────────────

static void test_row_sum_eager() {
    const i32 rows = 3, cols = 4;
    f32 x_data[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,  // row 0: sum = 10
        5.0f, 6.0f, 7.0f, 8.0f,  // row 1: sum = 26
        9.0f, 10.0f, 11.0f, 12.0f // row 2: sum = 42
    };
    f32 expected[3] = {10.0f, 26.0f, 42.0f};

    Matrix x(rows, cols), y(rows, 1);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    y = row_sum(x);

    f32 y_host[3];
    cudaMemcpy(y_host, y.data, 3 * sizeof(f32), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (i32 i = 0; i < 3; i++) {
        if (fabs(y_host[i] - expected[i]) > 1e-4f) {
            RLOG(LL_ERROR, "row_sum[%d] = %f, expected %f", i, y_host[i], expected[i]);
            ok = false;
        }
    }
    record(ok, "row_sum_eager");
}

// ─── Column sum tests ─────────────────────────────────────────────────────

static void test_col_sum_eager() {
    const i32 rows = 3, cols = 4;
    f32 x_data[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    // col sums: [1+5+9=15, 2+6+10=18, 3+7+11=21, 4+8+12=24]
    f32 expected[4] = {15.0f, 18.0f, 21.0f, 24.0f};

    Matrix x(rows, cols), y(1, cols);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    y = col_sum(x);

    f32 y_host[4];
    cudaMemcpy(y_host, y.data, 4 * sizeof(f32), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (i32 i = 0; i < 4; i++) {
        if (fabs(y_host[i] - expected[i]) > 1e-4f) {
            RLOG(LL_ERROR, "col_sum[%d] = %f, expected %f", i, y_host[i], expected[i]);
            ok = false;
        }
    }
    record(ok, "col_sum_eager");
}

// ─── Total sum tests ──────────────────────────────────────────────────────

static void test_sum_eager() {
    const i32 rows = 3, cols = 4;
    f32 x_data[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    // total sum = 10 + 26 + 42 = 78
    f32 expected = 78.0f;

    Matrix x(rows, cols), y(1, 1);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    y = sum(x);

    f32 y_host;
    cudaMemcpy(&y_host, y.data, sizeof(f32), cudaMemcpyDeviceToHost);

    bool ok = fabs(y_host - expected) < 1e-4f;
    if (!ok) {
        RLOG(LL_ERROR, "sum = %f, expected %f", y_host, expected);
    }
    record(ok, "sum_eager");
}

// ─── Out-param variants ───────────────────────────────────────────────────

static void test_row_sum_outparam() {
    const i32 rows = 2, cols = 3;
    f32 x_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    f32 expected[2] = {6.0f, 15.0f};

    Matrix x(rows, cols), y(rows, 1);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    row_sum(x, y);

    f32 y_host[2];
    cudaMemcpy(y_host, y.data, 2 * sizeof(f32), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (i32 i = 0; i < 2; i++) {
        if (fabs(y_host[i] - expected[i]) > 1e-4f) ok = false;
    }
    record(ok, "row_sum_outparam");
}

// ─── Edge cases ───────────────────────────────────────────────────────────

static void test_row_sum_single_row() {
    const i32 rows = 1, cols = 5;
    f32 x_data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    f32 expected = 15.0f;

    Matrix x(rows, cols), y(rows, 1);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    y = row_sum(x);

    f32 y_host;
    cudaMemcpy(&y_host, y.data, sizeof(f32), cudaMemcpyDeviceToHost);

    bool ok = fabs(y_host - expected) < 1e-4f;
    record(ok, "row_sum_single_row");
}

static void test_col_sum_single_col() {
    const i32 rows = 5, cols = 1;
    f32 x_data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    f32 expected = 15.0f;

    Matrix x(rows, cols), y(1, cols);
    cudaMemcpy(x.data, x_data, sizeof(x_data), cudaMemcpyHostToDevice);

    y = col_sum(x);

    f32 y_host;
    cudaMemcpy(&y_host, y.data, sizeof(f32), cudaMemcpyDeviceToHost);

    bool ok = fabs(y_host - expected) < 1e-4f;
    record(ok, "col_sum_single_col");
}

// ─── Main ─────────────────────────────────────────────────────────────────

int main() {
    initLog();
    test_row_sum_eager();
    test_col_sum_eager();
    test_sum_eager();
    test_row_sum_outparam();
    test_row_sum_single_row();
    test_col_sum_single_col();
    return testSummary();
}
