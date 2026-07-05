#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <nn/nn.cuh>
#include <harness_test.h>

#include <cmath>

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

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);
    test_relu_eager();
    test_relu_eager_v2();
    test_grad_relu();
    test_relu_outparam();
    return testSummary();
}
