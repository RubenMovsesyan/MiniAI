#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <fused/fused.cuh>
#include <nn/nn.cuh>
#include <harness_test.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// ─── Helpers ────────────────────────────────────────────────────────────────────

static void upload(Matrix& m, const f32* host) {
    cudaMemcpy(m.data, host, (usize)m.rows() * m.cols() * sizeof(f32), cudaMemcpyHostToDevice);
}

static void download(const Matrix& m, f32* host) {
    cudaMemcpy(host, m.data, (usize)m.rows() * m.cols() * sizeof(f32), cudaMemcpyDeviceToHost);
}

// The one host-readback point: sync the compute stream, copy the (1,1) loss to host.
static f32 lossScalar(const Matrix& loss) {
    cudaStreamSynchronize(g_compute_stream);
    f32 h;
    cudaMemcpy(&h, loss.data, sizeof(f32), cudaMemcpyDeviceToHost);
    return h;
}

// ─── CE correctness ─────────────────────────────────────────────────────────────

static void test_ce_uniform() {
    // Uniform logits over C classes → a2 = 1/C → CE = log(C) for any one-hot target.
    const i32 N = 1, C = 5;
    f32 logits[N * C] = {0, 0, 0, 0, 0};
    f32 targets[N * C] = {0, 0, 1, 0, 0};

    Matrix x(N, C), t(N, C);
    upload(x, logits); upload(t, targets);

    f32 ce = lossScalar(cross_entropy(x, t));
    bool ok = fabsf(ce - logf((f32)C)) < 1e-4f;
    if (!ok) RLOG(LL_ERROR, "CE uniform = %f, expected log(%d) = %f", ce, C, logf((f32)C));
    record(ok, "ce_uniform_logC");
}

static void test_ce_known() {
    // logits [1,2,3], target class 2 → CE = -log(softmax([1,2,3])[2]) = 0.40760595
    const i32 N = 1, C = 3;
    f32 logits[N * C] = {1.0f, 2.0f, 3.0f};
    f32 targets[N * C] = {0.0f, 0.0f, 1.0f};

    Matrix x(N, C), t(N, C);
    upload(x, logits); upload(t, targets);

    f32 ce = lossScalar(cross_entropy(x, t));
    bool ok = fabsf(ce - 0.40760595f) < 1e-4f;
    if (!ok) RLOG(LL_ERROR, "CE known = %f, expected 0.40760595", ce);
    record(ok, "ce_known_value");
}

// ─── Gradient check (central finite differences) ────────────────────────────────
// The single most important correctness gate: verify the fused analytic gradient
// (a2 - targets)/N matches the numerical gradient of cross_entropy element-by-element.

static void test_gradient_check() {
    const i32 N = 4, C = 3;
    const i32 n = N * C;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<f32> dist(-2.0f, 2.0f);
    std::uniform_int_distribution<i32> cls(0, C - 1);

    std::vector<f32> logits(n), targets(n, 0.0f);
    for (i32 i = 0; i < n; i++) logits[i] = dist(rng);
    for (i32 r = 0; r < N; r++) targets[r * C + cls(rng)] = 1.0f;  // one-hot per row

    Matrix x(N, C), t(N, C);
    upload(x, logits.data()); upload(t, targets.data());

    // Analytic gradient
    Matrix g = grad_softmax_cross_entropy(x, t);
    std::vector<f32> analytic(n);
    download(g, analytic.data());

    // Numerical gradient via central differences on cross_entropy
    const f32 h = 1e-3f;
    bool ok = true;
    for (i32 i = 0; i < n && ok; i++) {
        f32 saved = logits[i];

        logits[i] = saved + h; upload(x, logits.data());
        f32 lp = lossScalar(cross_entropy(x, t));

        logits[i] = saved - h; upload(x, logits.data());
        f32 lm = lossScalar(cross_entropy(x, t));

        logits[i] = saved;  // restore

        f32 numeric = (lp - lm) / (2.0f * h);
        if (fabsf(numeric - analytic[i]) > 1e-2f) {
            RLOG(LL_ERROR, "grad[%d]: analytic %f vs numeric %f", i, analytic[i], numeric);
            ok = false;
        }
    }
    record(ok, "gradient_check_finite_diff");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);
    test_ce_uniform();
    test_ce_known();
    test_gradient_check();
    return testSummary();
}
