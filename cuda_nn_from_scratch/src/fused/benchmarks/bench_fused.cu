#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <fused/fused.cuh>
#include <harness_bench.cuh>

#include <vector>

static const char* STATE = "src/fused/benchmarks/baseline.csv";
static const i32   ITERS = 50;

// ─── grad_softmax_cross_entropy benchmarks ──────────────────────────────────────

static void benchGradSoftmaxCE() {
    const i32 SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const i32 N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    static char lbl[N_SIZES][32];
    static const char* labels[N_SIZES + 1];

    std::vector<Matrix> xs, ts, outs;
    for (i32 i = 0; i < N_SIZES; i++) {
        i32 n = SIZES[i];
        xs.emplace_back(n, n);
        ts.emplace_back(n, n);
        outs.emplace_back(n, n);
        snprintf(lbl[i], sizeof(lbl[i]), "%dx%d", n, n);
        labels[i] = lbl[i];
    }
    labels[N_SIZES] = nullptr;

    BenchConfig cfg;
    cfg.name = "grad_softmax_cross_entropy"; cfg.labels = labels;
    cfg.state_csv = STATE; cfg.iterations = ITERS;
    benchmark(cfg, [](Matrix& out, const Matrix& x, const Matrix& t) {
        grad_softmax_cross_entropy(x, t, out);
    }, outs, xs, ts);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);
    benchGradSoftmaxCE();
    RLOG(LL_INFO, "fused benchmarks complete");
    return 0;
}
