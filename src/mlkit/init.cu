#include <rlog.h>
#include <mlkit/init.cuh>

#include <random>
#include <vector>
#include <cmath>

// Module-level RNG. Seeded from random_device on first use; mlkit_seed overrides
// for reproducibility. Host-side init is single-threaded, so no synchronization.
static std::mt19937 g_rng{std::random_device{}()};

void mlkit_seed(u32 seed) { g_rng.seed(seed); }

// ─── Distribution primitives ────────────────────────────────────────────────────

static void upload(Matrix& w, const std::vector<f32>& host) {
    cudaMemcpy(w.data, host.data(), host.size() * sizeof(f32), cudaMemcpyHostToDevice);
}

void fill_normal(Matrix& w, f32 std) {
    usize n = (usize)w.rows() * w.cols();
    std::vector<f32> host(n);
    std::normal_distribution<f32> dist(0.0f, std);
    for (usize i = 0; i < n; i++) host[i] = dist(g_rng);
    upload(w, host);
}

void fill_uniform(Matrix& w, f32 limit) {
    usize n = (usize)w.rows() * w.cols();
    std::vector<f32> host(n);
    std::uniform_real_distribution<f32> dist(-limit, limit);
    for (usize i = 0; i < n; i++) host[i] = dist(g_rng);
    upload(w, host);
}

void zero_init(Matrix& w) {
    cudaMemset(w.data, 0, (usize)w.rows() * w.cols() * sizeof(f32));  // 0.0f == all-zero bits
}

// ─── He (ReLU) ──────────────────────────────────────────────────────────────────

void he_normal (Matrix& w, i32 fan_in) { fill_normal (w, sqrtf(2.0f / (f32)fan_in)); }
void he_uniform(Matrix& w, i32 fan_in) { fill_uniform(w, sqrtf(6.0f / (f32)fan_in)); }
void he_normal (Matrix& w) { he_normal (w, w.rows()); }
void he_uniform(Matrix& w) { he_uniform(w, w.rows()); }

// ─── LeCun (linear, SELU) ────────────────────────────────────────────────────────

void lecun_normal (Matrix& w, i32 fan_in) { fill_normal (w, sqrtf(1.0f / (f32)fan_in)); }
void lecun_uniform(Matrix& w, i32 fan_in) { fill_uniform(w, sqrtf(3.0f / (f32)fan_in)); }
void lecun_normal (Matrix& w) { lecun_normal (w, w.rows()); }
void lecun_uniform(Matrix& w) { lecun_uniform(w, w.rows()); }

// ─── Xavier/Glorot (tanh, sigmoid) ───────────────────────────────────────────────

void xavier_normal(Matrix& w, i32 fan_in, i32 fan_out) {
    fill_normal(w, sqrtf(2.0f / (f32)(fan_in + fan_out)));
}
void xavier_uniform(Matrix& w, i32 fan_in, i32 fan_out) {
    fill_uniform(w, sqrtf(6.0f / (f32)(fan_in + fan_out)));
}
void xavier_normal (Matrix& w) { xavier_normal (w, w.rows(), w.cols()); }
void xavier_uniform(Matrix& w) { xavier_uniform(w, w.rows(), w.cols()); }
