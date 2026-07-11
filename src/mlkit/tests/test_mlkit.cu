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

// ─── Network engine tests ───────────────────────────────────────────────────────

static void upload(Matrix& m, const f32* host) {
    cudaMemcpy(m.data, host, (usize)m.rows() * m.cols() * sizeof(f32), cudaMemcpyHostToDevice);
}

// Linear forward, batched: Y = X·W + b computed per row.
static void test_linear_forward() {
    const i32 B = 2, IN = 3, OUT = 2;
    Dense d(B, IN, OUT, Activation::Identity, Init::He);
    f32 Xh[B*IN] = {1, 2, 3,  4, 5, 6};
    f32 Wh[IN*OUT] = {1, 0,  0, 1,  1, 1};   // (3×2)
    f32 bh[OUT] = {0.5f, -0.5f};
    Matrix X(B, IN);
    upload(X, Xh); upload(d.W, Wh); upload(d.b, bh);

    const Matrix& Y = d.forward(X);
    auto h = download(Y);
    // row0: [1+3, 2+3] + b = [4.5, 4.5]; row1: [4+6, 5+6] + b = [10.5, 10.5]
    f32 exp[B*OUT] = {4.5f, 4.5f, 10.5f, 10.5f};
    bool ok = true;
    for (i32 i = 0; i < B*OUT; i++) if (fabsf(h[i] - exp[i]) > 1e-4f) ok = false;
    record(ok, "linear_forward_batched");
}

// ReLU layer masks negatives.
static void test_relu_layer() {
    const i32 B = 1, IN = 3, OUT = 3;
    Dense d(B, IN, OUT, Activation::ReLU, Init::He);
    f32 Xh[3] = {-1, 2, -3};
    f32 Wh[9] = {1,0,0, 0,1,0, 0,0,1};   // identity
    f32 bh[3] = {0, 0, 0};
    Matrix X(B, IN);
    upload(X, Xh); upload(d.W, Wh); upload(d.b, bh);

    const Matrix& Y = d.forward(X);
    auto h = download(Y);
    f32 exp[3] = {0, 2, 0};
    bool ok = true;
    for (i32 i = 0; i < 3; i++) if (fabsf(h[i] - exp[i]) > 1e-4f) ok = false;
    record(ok, "relu_layer_forward");
}

// Finite-difference gradient check of Dense dW against the softmax-CE loss — the
// key correctness gate for the layer backward chain.
static void test_layer_backward_gradcheck() {
    const i32 B = 2, IN = 3, OUT = 2;
    Dense d(B, IN, OUT, Activation::Identity, Init::He);
    SoftmaxCrossEntropyLoss loss(B, OUT);

    f32 Xh[B*IN] = {0.5f, -1.0f, 2.0f,  -0.3f, 0.8f, 1.2f};
    f32 Wh[IN*OUT] = {0.1f, -0.2f, 0.3f, 0.05f, -0.4f, 0.15f};
    f32 bh[OUT] = {0.0f, 0.0f};
    f32 Yh[B*OUT] = {1, 0,  0, 1};    // one-hot targets
    Matrix X(B, IN), Y(B, OUT);
    upload(X, Xh); upload(Y, Yh); upload(d.W, Wh); upload(d.b, bh);

    // Analytic dW
    const Matrix& logits = d.forward(X);
    const Matrix& dA = loss.backward(logits, Y);
    d.zero_grad();
    d.backward(dA);
    auto analytic = download(d.dW);

    // Numerical dW via central differences of loss.value
    std::vector<f32> Wv(Wh, Wh + IN*OUT);
    const f32 hh = 1e-3f;
    bool ok = true;
    for (i32 i = 0; i < IN*OUT && ok; i++) {
        f32 saved = Wv[i];
        Wv[i] = saved + hh; upload(d.W, Wv.data());
        f32 lp = loss.value(d.forward(X), Y);
        Wv[i] = saved - hh; upload(d.W, Wv.data());
        f32 lm = loss.value(d.forward(X), Y);
        Wv[i] = saved;
        f32 numeric = (lp - lm) / (2.0f * hh);
        if (fabsf(numeric - analytic[i]) > 1e-2f) {
            RLOG(LL_ERROR, "dW[%d]: analytic %f vs numeric %f", i, analytic[i], numeric);
            ok = false;
        }
    }
    record(ok, "layer_backward_gradcheck");
}

// A tiny separable problem: loss must decrease over training steps.
static void test_learning() {
    const i32 B = 4, IN = 2, OUT = 2;
    mlkit_seed(2024);
    Network net = NetworkBuilder(B, IN)
        .dense(8, Activation::ReLU,     Init::He)
        .dense(OUT, Activation::Identity, Init::He)
        .loss_softmax_cross_entropy()
        .optimizer(std::make_unique<SGD>(0.1f))
        .eval_interval(1)
        .build();

    f32 Xh[B*IN] = {-1.0f, -1.0f,  -1.2f, -0.8f,   1.0f, 1.0f,   0.9f, 1.1f};
    f32 Yh[B*OUT] = {1, 0,  1, 0,   0, 1,   0, 1};
    Matrix X(B, IN), Y(B, OUT);
    upload(X, Xh); upload(Y, Yh);

    net.train_step(X, Y);
    f32 early = net.last_loss();
    for (i32 i = 0; i < 300; i++) net.train_step(X, Y);
    f32 late = net.last_loss();

    bool ok = late < early && late < 0.1f;
    if (!ok) RLOG(LL_ERROR, "learning: early %f late %f", early, late);
    record(ok, "learning_loss_decreases");
}

// eval_interval controls readback cadence; 0 = never.
static void test_eval_interval() {
    const i32 B = 4, IN = 2, OUT = 2;
    mlkit_seed(5);
    Network net = NetworkBuilder(B, IN)
        .dense(OUT, Activation::Identity, Init::He)
        .loss_softmax_cross_entropy()
        .optimizer(std::make_unique<SGD>(0.05f))
        .eval_interval(3)
        .build();

    f32 Xh[B*IN] = {-1.0f, -1.0f,  -1.2f, -0.8f,   1.0f, 1.0f,   0.9f, 1.1f};
    f32 Yh[B*OUT] = {1, 0,  1, 0,   0, 1,   0, 1};
    Matrix X(B, IN), Y(B, OUT);
    upload(X, Xh); upload(Y, Yh);

    net.train_step(X, Y);                       // step 1
    net.train_step(X, Y);                       // step 2 — no readback yet
    bool before = (net.last_loss() == 0.0f);
    net.train_step(X, Y);                       // step 3 — readback fires
    bool after = (net.last_loss() != 0.0f);
    record(before && after, "eval_interval_cadence");
}

// Builder rejects an activation on the final layer (double-softmax guard).
static void test_double_softmax_guard() {
    NetworkBuilder bad(4, 2);
    bad.dense(8, Activation::ReLU, Init::He).dense(2, Activation::ReLU, Init::He);  // final ReLU: bad
    NetworkBuilder good(4, 2);
    good.dense(8, Activation::ReLU, Init::He).dense(2, Activation::Identity, Init::He);
    record(!bad.output_layer_ok() && good.output_layer_ok(), "double_softmax_guard");
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

    test_linear_forward();
    test_relu_layer();
    test_layer_backward_gradcheck();
    test_learning();
    test_eval_interval();
    test_double_softmax_guard();
    return testSummary();
}
