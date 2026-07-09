#pragma once

#include <matrix/matrix.cuh>

// ─── Weight initialization ──────────────────────────────────────────────────────
// Variance-scaled init keeps activation variance ~constant through depth, avoiding
// the exploding/vanishing gradients that flat ±0.5 init causes. In-place fill on a
// preallocated Matrix; host-side RNG + a single host→device upload (no kernel).
//
// Shape-derive overloads assume the layout W = fan_in × fan_out
// (fan_in = W.rows(), fan_out = W.cols()).
//
// Scheme → activation family:
//   He     — ReLU:                var = 2/fan_in
//   LeCun  — linear, SELU:        var = 1/fan_in
//   Xavier — symmetric (tanh, sigmoid): var = 2/(fan_in+fan_out)

// RNG control — module-level mt19937 (seeded from random_device on first use).
// Call before an init for reproducible weights (tests rely on this).
void mlkit_seed(u32 seed);

// Distribution primitives
void fill_normal (Matrix& w, f32 std);     // N(0, std^2)
void fill_uniform(Matrix& w, f32 limit);   // U(-limit, +limit)
void zero_init   (Matrix& w);              // biases → 0

// He (ReLU): std = √(2/fan_in), limit = √(6/fan_in)
void he_normal (Matrix& w, i32 fan_in);   void he_normal (Matrix& w);
void he_uniform(Matrix& w, i32 fan_in);   void he_uniform(Matrix& w);

// LeCun (linear, SELU): std = √(1/fan_in), limit = √(3/fan_in)
void lecun_normal (Matrix& w, i32 fan_in);  void lecun_normal (Matrix& w);
void lecun_uniform(Matrix& w, i32 fan_in);  void lecun_uniform(Matrix& w);

// Xavier/Glorot (tanh, sigmoid): std = √(2/(fan_in+fan_out)), limit = √(6/(fan_in+fan_out))
void xavier_normal (Matrix& w, i32 fan_in, i32 fan_out);  void xavier_normal (Matrix& w);
void xavier_uniform(Matrix& w, i32 fan_in, i32 fan_out);  void xavier_uniform(Matrix& w);
