#pragma once

#include <matrix/matrix.cuh>
#include <mlkit/optimizer.cuh>

// ─── Layout convention (row-major, enforced everywhere) ─────────────────────────
//   batch = ROWS. Forward:  Y = X·W + b
//     X : (B × in)      activations, one sample per row
//     W : (in × out)    weights, layout W = fan_in × fan_out (matches mlkit init)
//     b : (1 × out)     bias, broadcast across the B rows via rowAdd
//   Linear backward (upstream dZ is the gradient w.r.t. this layer's pre-activation):
//     dW = Xᵀ·dZ    (in × out)   accumulated
//     db = col_sum(dZ)  (1 × out) — sum over the batch rows, accumulated
//     dX = dZ·Wᵀ    (B × in)     handed to the previous layer
//   The 1/N (batch mean) lives ONLY in the loss gradient; layers just propagate.

enum class Activation { ReLU, Identity };
enum class Init       { He, LeCun, Xavier };

// ─── Layer base ─────────────────────────────────────────────────────────────────
// Abstract interface so new layer types (conv, dropout, …) drop in without touching
// Network. Dense is the only implementation today. Buffers are preallocated so a
// training step allocates nothing on the hot path (aside from matmul materialization
// temps). A parameter-free layer implements zero_grad/update as no-ops.

struct Layer {
    virtual ~Layer() = default;
    virtual const Matrix& forward(const Matrix& X) = 0;   // returns this layer's output
    virtual const Matrix& backward(const Matrix& dA) = 0; // returns dX for the previous layer
    virtual void zero_grad() = 0;
    virtual void update(Optimizer& opt) = 0;
    // For the builder's double-softmax guard: does this layer apply a non-identity
    // activation? The final layer must NOT (softmax lives in the loss).
    virtual bool has_activation() const { return false; }
};

// ─── Dense (fully-connected) layer ──────────────────────────────────────────────

struct Dense : Layer {
    Matrix W, b, dW, db;          // parameters + accumulated gradients
    Matrix Z, A, dZ, dX;          // preact, postact, local grad, input-grad (all B×·)
    // Scratch, preallocated so the hot path never calls cudaMalloc/cudaFree (both are
    // device-synchronizing — allocating per step would stall the async stream).
    Matrix Xt;                    // (in × B)   Xᵀ
    Matrix Wt;                    // (out × in) Wᵀ
    Matrix dW_grad;               // (in × out) this step's Xᵀ·dZ before accumulation
    Matrix db_grad;               // (1 × out)  this step's col_sum(dZ)
    Activation act;
    const Matrix* input_cache = nullptr;  // input to the last forward (owned upstream)

    Dense(i32 batch, i32 in, i32 out, Activation activation, Init init);

    const Matrix& forward(const Matrix& X) override;
    const Matrix& backward(const Matrix& dA) override;
    void zero_grad() override;
    void update(Optimizer& opt) override;
    bool has_activation() const override { return act != Activation::Identity; }
};
