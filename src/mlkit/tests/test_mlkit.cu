#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <mlkit/mlkit.cuh>
#include <harness_test.h>

#include <cmath>
#include <vector>

// ─── Helpers ────────────────────────────────────────────────────────────────────

static std::vector<f32> download(const Matrix& m) {
    usize n = (usize)m.rows() * m.cols();
    std::vector<f32> h(n);
    cudaMemcpy(h.data(), m.data, n * sizeof(f32), cudaMemcpyDeviceToHost);
    return h;
}

static f32 mean_of(const std::vector<f32>& v) {
    f64 s = 0;
    for (f32 x : v) s += x;
    return (f32)(s / v.size());
}

static f32 std_of(const std::vector<f32>& v) {
    f32 m = mean_of(v);
    f64 s = 0;
    for (f32 x : v) s += (f64)(x - m) * (x - m);
    return (f32)sqrt(s / v.size());
}

static bool close_rel(f32 got, f32 expected, f32 rel) {
    return fabsf(got - expected) <= rel * expected;
}

// ─── Normal-scheme std checks (mean ≈ 0, std ≈ target) ───────────────────────────

static void test_normal_std(const char* label, void (*init)(Matrix&, i32), i32 fan, f32 expected_std) {
    Matrix w(512, 512);
    mlkit_seed(123);
    init(w, fan);
    auto h = download(w);
    f32 mean = mean_of(h), sd = std_of(h);
    bool ok = fabsf(mean) < 0.005f && close_rel(sd, expected_std, 0.03f);
    if (!ok) RLOG(LL_ERROR, "%s: mean %f, std %f (expected std %f)", label, mean, sd, expected_std);
    record(ok, label);
}

static void test_xavier_normal_std() {
    Matrix w(512, 512);
    mlkit_seed(123);
    xavier_normal(w, 784, 128);
    auto h = download(w);
    f32 expected = sqrtf(2.0f / (784.0f + 128.0f));
    f32 mean = mean_of(h), sd = std_of(h);
    bool ok = fabsf(mean) < 0.005f && close_rel(sd, expected, 0.03f);
    if (!ok) RLOG(LL_ERROR, "xavier_normal: mean %f std %f (expected %f)", mean, sd, expected);
    record(ok, "xavier_normal_std");
}

// ─── Uniform: in-range + std ≈ limit/√3 ──────────────────────────────────────────

static void test_he_uniform_range() {
    Matrix w(512, 512);
    mlkit_seed(123);
    he_uniform(w, 784);
    auto h = download(w);
    f32 limit = sqrtf(6.0f / 784.0f);
    bool in_range = true;
    for (f32 x : h) if (x < -limit || x > limit) { in_range = false; break; }
    f32 sd = std_of(h);
    bool ok = in_range && close_rel(sd, limit / sqrtf(3.0f), 0.03f);
    if (!ok) RLOG(LL_ERROR, "he_uniform: in_range %d std %f (expected %f)", in_range, sd, limit / sqrtf(3.0f));
    record(ok, "he_uniform_range_std");
}

// ─── zero_init ───────────────────────────────────────────────────────────────────

static void test_zero_init() {
    Matrix w(64, 64);
    he_normal(w, 64);       // dirty it first
    zero_init(w);
    auto h = download(w);
    bool ok = true;
    for (f32 x : h) if (x != 0.0f) { ok = false; break; }
    record(ok, "zero_init");
}

// ─── Determinism + derive-equals-explicit ────────────────────────────────────────

static bool buffers_equal(const Matrix& a, const Matrix& b) {
    auto ha = download(a), hb = download(b);
    if (ha.size() != hb.size()) return false;
    for (usize i = 0; i < ha.size(); i++) if (ha[i] != hb[i]) return false;
    return true;
}

static void test_determinism() {
    Matrix a(128, 64), b(128, 64);
    mlkit_seed(7); he_normal(a, 100);
    mlkit_seed(7); he_normal(b, 100);
    record(buffers_equal(a, b), "determinism_same_seed");
}

static void test_derive_equals_explicit() {
    Matrix a(128, 64), b(128, 64);
    mlkit_seed(7); he_normal(a, a.rows());   // explicit fan_in = rows
    mlkit_seed(7); he_normal(b);             // derive fan_in = rows
    bool he_ok = buffers_equal(a, b);

    Matrix c(128, 64), d(128, 64);
    mlkit_seed(9); xavier_normal(c, c.rows(), c.cols());
    mlkit_seed(9); xavier_normal(d);
    bool xav_ok = buffers_equal(c, d);

    record(he_ok && xav_ok, "derive_equals_explicit");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);
    test_normal_std("he_normal_std",    he_normal,    784, sqrtf(2.0f / 784.0f));  // ≈0.0505
    test_normal_std("lecun_normal_std", lecun_normal, 784, sqrtf(1.0f / 784.0f));  // ≈0.0357
    test_xavier_normal_std();
    test_he_uniform_range();
    test_zero_init();
    test_determinism();
    test_derive_equals_explicit();
    return testSummary();
}
