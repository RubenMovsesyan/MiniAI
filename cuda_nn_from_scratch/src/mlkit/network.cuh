#pragma once

#include <matrix/matrix.cuh>
#include <mlkit/layer.cuh>
#include <mlkit/loss.cuh>
#include <mlkit/optimizer.cuh>

#include <memory>
#include <vector>

// ─── Network ────────────────────────────────────────────────────────────────────
// A built, ready-to-train engine: an ordered stack of Layers + a softmax-CE loss +
// an optimizer, all preallocated. Everything stays on-device across a step; the only
// host transfer is the deliberate loss readback every `eval_interval` steps.

class Network {
public:
    Network(std::vector<std::unique_ptr<Layer>> layers, i32 batch, i32 classes,
            std::unique_ptr<Optimizer> opt, i32 eval_interval);

    // Inference: run the stack, return logits (caller can softmax for predictions).
    const Matrix& forward(const Matrix& X);

    // One training step on a batch: forward → loss.backward → zero_grad → backward
    // (accumulate) → optimizer update. Every `eval_interval` steps (0 = never) the
    // loss is copied to host, retrievable via last_loss().
    void train_step(const Matrix& X, const Matrix& Y);

    f32 last_loss() const { return last_loss_; }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    SoftmaxCrossEntropyLoss             loss_;
    std::unique_ptr<Optimizer>          opt_;
    i32 eval_interval_;
    u64 step_count_ = 0;
    f32 last_loss_  = 0.0f;
};

// ─── NetworkBuilder (factory) ───────────────────────────────────────────────────
// Fluent factory. Records layer specs, then build() allocates buffers, initializes
// weights, wires the loss + optimizer, and returns a ready Network. Extend with new
// layer kinds (e.g. .conv(...)) alongside .dense(...) as they are implemented.

class NetworkBuilder {
public:
    NetworkBuilder(i32 batch, i32 input_dim)
        : batch_(batch), cur_dim_(input_dim) {}

    NetworkBuilder& dense(i32 units, Activation act, Init init);
    NetworkBuilder& loss_softmax_cross_entropy() { return *this; }  // only loss type today
    NetworkBuilder& optimizer(std::unique_ptr<Optimizer> o) { opt_ = std::move(o); return *this; }
    NetworkBuilder& eval_interval(i32 n) { eval_interval_ = n; return *this; }

    // Double-softmax guard: the final layer must emit logits (Identity activation).
    bool output_layer_ok() const {
        return !layers_.empty() && !layers_.back()->has_activation();
    }

    Network build();

private:
    i32 batch_;
    i32 cur_dim_;
    i32 classes_ = 0;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<Optimizer> opt_;
    i32 eval_interval_ = 0;
};
