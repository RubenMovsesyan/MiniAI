#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <harness_bench.cuh>
#include <harness_matrix_csv.cuh>

#include <vector>

static const char* STATE = "src/matrix/benchmarks/baseline.csv";
static const i32   ITERS = 50;

// ─── Input loading ────────────────────────────────────────────────────────────

// Load one CSV (A/B/col/row) for every variant into a list of GPU matrices.
static std::vector<Matrix> loadList(const char** variants, const char* file) {
    std::vector<Matrix> v;
    for (i32 i = 0; variants[i]; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/inputs/%s/%s.csv", DATA_ROOT, variants[i], file);
        v.push_back(matLoad(path));
    }
    return v;
}

// Output buffers matching input dims (element-wise ops).
static std::vector<Matrix> makeOuts(const std::vector<Matrix>& ref) {
    std::vector<Matrix> v;
    for (const auto& m : ref) v.emplace_back(m.rows(), m.cols());
    return v;
}

// Output buffers with swapped dims (transpose).
static std::vector<Matrix> makeOutsT(const std::vector<Matrix>& ref) {
    std::vector<Matrix> v;
    for (const auto& m : ref) v.emplace_back(m.cols(), m.rows());
    return v;
}

static BenchConfig baseCfg(const char* name, const char** labels) {
    BenchConfig c;
    c.name = name; c.labels = labels; c.state_csv = STATE; c.iterations = ITERS;
    return c;
}

// ─── Element-wise groups (all variants) ───────────────────────────────────────

template <typename Op>
static void benchAB(const char* name, Op op) {
    auto As = loadList(ALL_VARIANTS, "A");
    auto Bs = loadList(ALL_VARIANTS, "B");
    auto Cs = makeOuts(As);
    BenchConfig cfg = baseCfg(name, ALL_VARIANTS);
    benchmark(cfg, op, Cs, As, Bs);
}

static void benchScalar() {
    auto As = loadList(ALL_VARIANTS, "A");
    auto Cs = makeOuts(As);
    BenchConfig cfg = baseCfg("scalar", ALL_VARIANTS);
    benchmark(cfg, [](Matrix& C, const Matrix& A) { C = A * 2.0f; }, Cs, As);
}

static void benchTranspose() {
    auto As = loadList(ALL_VARIANTS, "A");
    auto Cs = makeOutsT(As);
    BenchConfig cfg = baseCfg("transpose", ALL_VARIANTS);
    benchmark(cfg, [](Matrix& C, const Matrix& A) { C = A.lazy().transpose(); }, Cs, As);
}

static void benchColAdd() {
    auto As   = loadList(ALL_VARIANTS, "A");
    auto cols = loadList(ALL_VARIANTS, "col");
    auto Cs   = makeOuts(As);
    BenchConfig cfg = baseCfg("colAdd", ALL_VARIANTS);
    benchmark(cfg, [](Matrix& C, const Matrix& A, const Matrix& col) { C = A.lazy().colAdd(col); }, Cs, As, cols);
}

static void benchRowAdd() {
    auto As   = loadList(ALL_VARIANTS, "A");
    auto rows = loadList(ALL_VARIANTS, "row");
    auto Cs   = makeOuts(As);
    BenchConfig cfg = baseCfg("rowAdd", ALL_VARIANTS);
    benchmark(cfg, [](Matrix& C, const Matrix& A, const Matrix& row) { C = A.lazy().rowAdd(row); }, Cs, As, rows);
}

// ─── Matmul groups (square variants) ──────────────────────────────────────────

template <typename Op>
static void benchMM(const char* name, Op op) {
    auto As = loadList(SQUARE_VARIANTS, "A");
    auto Bs = loadList(SQUARE_VARIANTS, "B");
    auto Cs = makeOuts(As);
    BenchConfig cfg = baseCfg(name, SQUARE_VARIANTS);
    benchmark(cfg, op, Cs, As, Bs);
}

// ─── Eager (runtime) API — materialize-per-step path (square only) ────────────
// Contrast against the fused matmul/chain groups above: eager chains allocate an
// intermediate Matrix per step instead of fusing.

static void benchEager() {
    auto As = loadList(SQUARE_VARIANTS, "A");
    auto Bs = loadList(SQUARE_VARIANTS, "B");
    auto Cs = makeOuts(As);
    BenchConfig cfg1 = baseCfg("eager_matmul_chain", SQUARE_VARIANTS);
    benchmark(cfg1, [](Matrix& C, const Matrix& A, const Matrix& B) { C = A.matmul(B).matmul(A); }, Cs, As, Bs);
    BenchConfig cfg2 = baseCfg("eager_add_scale", SQUARE_VARIANTS);
    benchmark(cfg2, [](Matrix& C, const Matrix& A, const Matrix& B) { C = A.add(B).scale(2.0f); }, Cs, As, Bs);
}

// ─── Large square matmul (device-generated; no CSV) ───────────────────────────
// GEMM timing is value-independent, so big sizes are filled on-device instead of
// loaded from multi-GB CSVs. Correctness is covered by the <=1024 CSV tests.

__global__ static void fillKernel(f32* d, i32 n, f32 base) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = base + (i & 7) * 0.125f;  // nonzero, varied
}

static Matrix makeBig(i32 n) {
    Matrix m(n, n);
    i32 cnt = n * n;
    fillKernel<<<(cnt + 255) / 256, 256>>>(m.data, cnt, 1.0f);
    cudaDeviceSynchronize();
    return m;
}

static const i32 BIG_SIZES[] = {512, 1024, 2048, 4096, 8192, 10240};

static void benchBigMatmul() {
    constexpr i32 NS = sizeof(BIG_SIZES) / sizeof(BIG_SIZES[0]);
    static char lbl[NS][16];
    static const char* labels[NS + 1];
    std::vector<Matrix> As, Bs, Cs;
    for (i32 i = 0; i < NS; i++) {
        As.push_back(makeBig(BIG_SIZES[i]));
        Bs.push_back(makeBig(BIG_SIZES[i]));
        Cs.emplace_back(BIG_SIZES[i], BIG_SIZES[i]);
        snprintf(lbl[i], sizeof(lbl[i]), "%dx%d", BIG_SIZES[i], BIG_SIZES[i]);
        labels[i] = lbl[i];
    }
    labels[NS] = nullptr;

    BenchConfig cfg;
    cfg.name = "matmul_big"; cfg.labels = labels; cfg.state_csv = STATE; cfg.iterations = 20;
    benchmark(cfg, [](Matrix& C, const Matrix& A, const Matrix& B) { C = A * B; }, Cs, As, Bs);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);

    benchAB("add",      [](Matrix& C, const Matrix& A, const Matrix& B) { C = A + B; });
    benchAB("sub",      [](Matrix& C, const Matrix& A, const Matrix& B) { C = A - B; });
    benchAB("hadamard", [](Matrix& C, const Matrix& A, const Matrix& B) { C = A.lazy().hadamard(B); });
    benchScalar();
    benchTranspose();
    benchColAdd();
    benchRowAdd();

    benchMM("matmul",            [](Matrix& C, const Matrix& A, const Matrix& B) { C = A * B; });
    benchMM("chain_matmul_add",  [](Matrix& C, const Matrix& A, const Matrix& B) { C = A * B + A; });
    benchMM("chain_matmul_scale",[](Matrix& C, const Matrix& A, const Matrix& B) { C = (A * B) * 2.0f; });
    benchMM("chain_add_matmul",  [](Matrix& C, const Matrix& A, const Matrix& B) { C = (A + B) * A; });

    benchEager();
    benchBigMatmul();

    RLOG(LL_INFO, "benchmarks complete");
    return 0;
}
