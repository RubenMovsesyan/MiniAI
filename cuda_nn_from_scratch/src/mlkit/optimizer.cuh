#pragma once

#include <matrix/matrix.cuh>

// ─── Optimizer ──────────────────────────────────────────────────────────────────
// Pluggable weight-update strategy. update() is called once per parameter per step
// with the parameter and its (accumulated) gradient. SGD is the only implementation
// for now; momentum/Adam would add per-parameter state (register params at build).

struct Optimizer {
    virtual ~Optimizer() = default;
    virtual void update(Matrix& param, const Matrix& grad) = 0;
};

struct SGD : Optimizer {
    f32 lr;
    explicit SGD(f32 learning_rate) : lr(learning_rate) {}
    // param -= lr * grad — one fused element-wise kernel.
    void update(Matrix& p, const Matrix& g) override { p = p.ref() - g.ref() * lr; }
};
