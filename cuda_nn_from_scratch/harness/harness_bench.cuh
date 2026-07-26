#pragma once

#include <common.h>
#include <rlog.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

// ─── CUDA error check ─────────────────────────────────────────────────────────
#define CUDA_CHECK(x) harnessCudaCheck((x), #x, __FILE__, __LINE__)
inline void harnessCudaCheck(cudaError_t e, const char* call, const char* file, i32 line) {
    if (e != cudaSuccess) {
        RLOG(LL_FATAL, "CUDA error %s:%d  %s -> %s", file, line, call, cudaGetErrorString(e));
        exit(1);
    }
}

// ─── Config ───────────────────────────────────────────────────────────────────
enum class BenchMode { Sequential, RoundRobin };

struct BenchConfig {
    const char*  name       = "bench";   // group name; key prefix in the state CSV
    i32          iterations = 50;
    BenchMode    mode       = BenchMode::Sequential;
    f32          epsilon    = 0.10f;     // >10% slower than baseline → WARN
    const char*  state_csv  = nullptr;   // baseline file, e.g. "src/matrix/benchmarks/baseline.csv"
    const char** labels     = nullptr;   // per-row label (>= rows entries); used in output + baseline key
};

// ─── L2 cache flusher ─────────────────────────────────────────────────────────
// Memset a buffer >= L2 size between timed iterations so reads aren't artificially hot.
struct L2Flusher {
    void* buf = nullptr;
    usize size = 0;
    L2Flusher() {
        i32 dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
        i32 l2 = 0; CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
        size = (usize)l2 * 2;
        if (size == 0) size = 8 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&buf, size));
    }
    ~L2Flusher() { if (buf) cudaFree(buf); }
    void flush() { CUDA_CHECK(cudaMemset(buf, 0, size)); }
};

// ─── Stats ────────────────────────────────────────────────────────────────────
struct BenchStat { f32 min, median, mean, max, stddev; };

inline BenchStat computeStat(std::vector<f32>& t) {
    std::sort(t.begin(), t.end());
    BenchStat s{};
    i32 n = (i32)t.size();
    s.min = t.front();
    s.max = t.back();
    s.median = (n % 2) ? t[n / 2] : 0.5f * (t[n / 2 - 1] + t[n / 2]);
    f32 sum = 0.0f; for (f32 v : t) sum += v;
    s.mean = sum / n;
    f32 acc = 0.0f; for (f32 v : t) acc += (v - s.mean) * (v - s.mean);
    s.stddev = sqrtf(acc / n);
    return s;
}

// ─── Baseline state CSV ───────────────────────────────────────────────────────
// One line per row: "name,label,median_ms". Key is "name/label".
inline std::map<std::string, f32> loadBaselines(const char* path) {
    std::map<std::string, f32> m;
    if (!path) return m;
    FILE* f = fopen(path, "r");
    if (!f) return m;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char* n = strtok(line, ",\n\r");
        char* l = strtok(nullptr, ",\n\r");
        char* v = strtok(nullptr, ",\n\r");
        if (n && l && v) m[std::string(n) + "/" + l] = (f32)atof(v);
    }
    fclose(f);
    return m;
}

inline void writeBaselines(const char* path, const std::map<std::string, f32>& m) {
    if (!path) return;
    FILE* f = fopen(path, "w");
    if (!f) { RLOG(LL_WARN, "bench: cannot write baseline %s", path); return; }
    char line[1024];
    for (const auto& kv : m) {
        usize slash = kv.first.rfind('/');
        std::string name  = kv.first.substr(0, slash);
        std::string label = kv.first.substr(slash + 1);
        i32 len = snprintf(line, sizeof(line), "%s,%s,%.6f\n",
                           name.c_str(), label.c_str(), (double)kv.second);
        fwrite(line, 1, (usize)len, f);
    }
    fclose(f);
}

// ─── Entry point ──────────────────────────────────────────────────────────────
// Time `func` over rows of per-parameter lists. func is called as func(lists[i]...),
// performing the full op (incl. its own output buffer) inside the timed region.
//
// ponytail: the timed region includes Matrix::operator='s internal cudaDeviceSynchronize
// and any matmul temp allocations — intentional, we benchmark the public op not a bare kernel.
template <typename First, typename... Rest>
inline usize benchFirstSize(First& f, Rest&...) { return f.size(); }

template <typename F, typename... Lists>
void benchmark(const BenchConfig& cfg, F&& func, Lists&... lists) {
    usize rows = benchFirstSize(lists...);
    bool sizesOk = ((lists.size() == rows) && ...);
    if (!sizesOk) { RLOG(LL_ERROR, "bench %s: input lists differ in length", cfg.name); return; }
    if (rows == 0) return;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    L2Flusher flusher;

    std::vector<std::vector<f32>> times(rows);
    for (auto& v : times) v.reserve(cfg.iterations);

    auto run = [&](usize i) { func(lists[i]...); };
    auto timed = [&](usize i) {
        flusher.flush();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start, 0));
        run(i);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        f32 ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i].push_back(ms);
    };

    if (cfg.mode == BenchMode::Sequential) {
        for (usize i = 0; i < rows; i++) {
            run(i);  // warmup
            CUDA_CHECK(cudaDeviceSynchronize());
            for (i32 it = 0; it < cfg.iterations; it++) timed(i);
        }
    } else {  // RoundRobin: all rows resident, interleave
        for (usize i = 0; i < rows; i++) run(i);  // warmup all
        CUDA_CHECK(cudaDeviceSynchronize());
        for (i32 it = 0; it < cfg.iterations; it++)
            for (usize i = 0; i < rows; i++) timed(i);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    auto baselines = loadBaselines(cfg.state_csv);
    bool rebaseline = getenv("BENCH_REBASELINE") != nullptr;

    for (usize i = 0; i < rows; i++) {
        BenchStat s = computeStat(times[i]);
        char ibuf[32];
        const char* label = cfg.labels ? cfg.labels[i] : (snprintf(ibuf, sizeof(ibuf), "%zu", i), ibuf);
        RLOG(LL_INFO, "%s/%s: median %.4f ms (min %.4f, max %.4f, sigma %.4f)",
             cfg.name, label, (double)s.median, (double)s.min, (double)s.max, (double)s.stddev);

        std::string key = std::string(cfg.name) + "/" + label;
        auto it = baselines.find(key);
        if (it == baselines.end() || rebaseline) {
            baselines[key] = s.median;
            if (it == baselines.end())
                RLOG(LL_INFO, "  baseline set %.4f ms", (double)s.median);
        } else {
            f32 base = it->second;
            f32 ratio = s.median / base;
            if (s.median > base * (1.0f + cfg.epsilon))
                RLOG(LL_WARN, "  SLOWDOWN %s: baseline %.4f ms now %.4f ms (+%.1f%%)",
                     key.c_str(), (double)base, (double)s.median, (double)((ratio - 1.0f) * 100.0f));
            else if (s.median < base * (1.0f - cfg.epsilon))
                RLOG(LL_INFO, "  IMPROVED %s: baseline %.4f ms now %.4f ms (-%.1f%%)",
                     key.c_str(), (double)base, (double)s.median, (double)((1.0f - ratio) * 100.0f));
        }
    }

    writeBaselines(cfg.state_csv, baselines);
}
