#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <agg/agg.cuh>
#include <harness_bench.cuh>

#include <vector>

static const char* STATE = "src/agg/benchmarks/baseline.csv";
static const i32   ITERS = 50;

// ─── Row sum benchmarks ────────────────────────────────────────────────────

static void benchRowSum() {
    const i32 SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const i32 N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    static char lbl[N_SIZES][32];
    static const char* labels[N_SIZES + 1];

    std::vector<Matrix> xs, ys;
    for (i32 i = 0; i < N_SIZES; i++) {
        i32 n = SIZES[i];
        xs.emplace_back(n, n);
        ys.emplace_back(n, 1);
        snprintf(lbl[i], sizeof(lbl[i]), "%dx%d", n, n);
        labels[i] = lbl[i];
    }
    labels[N_SIZES] = nullptr;

    BenchConfig cfg;
    cfg.name = "row_sum"; cfg.labels = labels; cfg.state_csv = STATE; cfg.iterations = ITERS;
    benchmark(cfg, [](Matrix& y, const Matrix& x) { y = row_sum(x); }, ys, xs);
}

// ─── Column sum benchmarks ────────────────────────────────────────────────

static void benchColSum() {
    const i32 SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const i32 N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    static char lbl[N_SIZES][32];
    static const char* labels[N_SIZES + 1];

    std::vector<Matrix> xs, ys;
    for (i32 i = 0; i < N_SIZES; i++) {
        i32 n = SIZES[i];
        xs.emplace_back(n, n);
        ys.emplace_back(1, n);
        snprintf(lbl[i], sizeof(lbl[i]), "%dx%d", n, n);
        labels[i] = lbl[i];
    }
    labels[N_SIZES] = nullptr;

    BenchConfig cfg;
    cfg.name = "col_sum"; cfg.labels = labels; cfg.state_csv = STATE; cfg.iterations = ITERS;
    benchmark(cfg, [](Matrix& y, const Matrix& x) { y = col_sum(x); }, ys, xs);
}

// ─── Total sum benchmarks ─────────────────────────────────────────────────

static void benchSum() {
    const i32 SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const i32 N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    static char lbl[N_SIZES][32];
    static const char* labels[N_SIZES + 1];

    std::vector<Matrix> xs, ys;
    for (i32 i = 0; i < N_SIZES; i++) {
        i32 n = SIZES[i];
        xs.emplace_back(n, n);
        ys.emplace_back(1, 1);
        snprintf(lbl[i], sizeof(lbl[i]), "%dx%d", n, n);
        labels[i] = lbl[i];
    }
    labels[N_SIZES] = nullptr;

    BenchConfig cfg;
    cfg.name = "sum"; cfg.labels = labels; cfg.state_csv = STATE; cfg.iterations = ITERS;
    benchmark(cfg, [](Matrix& y, const Matrix& x) { y = sum(x); }, ys, xs);
}

// ─── Main ─────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);
    benchRowSum();
    benchColSum();
    benchSum();
    RLOG(LL_INFO, "aggregation benchmarks complete");
    return 0;
}
