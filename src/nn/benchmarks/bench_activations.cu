#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <nn/nn.cuh>
#include <harness_bench.cuh>

#include <vector>

static const char* STATE = "src/nn/benchmarks/baseline.csv";
static const i32   ITERS = 50;

// ─── Relu benchmarks ──────────────────────────────────────────────────────────

static void benchRelu() {
    const i32 SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const i32 N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    static char lbl[N_SIZES][32];
    static const char* labels[N_SIZES + 1];

    std::vector<Matrix> xs, ys;
    for (i32 i = 0; i < N_SIZES; i++) {
        i32 n = SIZES[i];
        xs.emplace_back(n, n);
        ys.emplace_back(n, n);
        snprintf(lbl[i], sizeof(lbl[i]), "%dx%d", n, n);
        labels[i] = lbl[i];
    }
    labels[N_SIZES] = nullptr;

    BenchConfig cfg;
    cfg.name = "relu"; cfg.labels = labels; cfg.state_csv = STATE; cfg.iterations = ITERS;
    benchmark(cfg, [](Matrix& y, const Matrix& x) { y = relu(x); }, ys, xs);
}

// ─── Grad relu benchmarks ─────────────────────────────────────────────────────

static void benchGradRelu() {
    const i32 SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const i32 N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    static char lbl[N_SIZES][32];
    static const char* labels[N_SIZES + 1];

    std::vector<Matrix> xs, dys, dxs;
    for (i32 i = 0; i < N_SIZES; i++) {
        i32 n = SIZES[i];
        xs.emplace_back(n, n);
        dys.emplace_back(n, n);
        dxs.emplace_back(n, n);
        snprintf(lbl[i], sizeof(lbl[i]), "%dx%d", n, n);
        labels[i] = lbl[i];
    }
    labels[N_SIZES] = nullptr;

    BenchConfig cfg;
    cfg.name = "grad_relu"; cfg.labels = labels; cfg.state_csv = STATE; cfg.iterations = ITERS;
    benchmark(cfg, [](Matrix& dx, const Matrix& x, const Matrix& dy) { dx = grad_relu(x, dy); }, dxs, xs, dys);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog();
    benchRelu();
    benchGradRelu();
    RLOG(LL_INFO, "activation benchmarks complete");
    return 0;
}
