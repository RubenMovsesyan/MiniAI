#pragma once

#include <matrix/matrix.cuh>

// ─── Dataset ────────────────────────────────────────────────────────────────────
// Format-agnostic: wraps any (X, Y) pair already on the GPU, whatever loaded it.
// Layout follows the engine convention — batch = rows:
//   X : (N × features)   Y : (N × classes)
//
// Rows stay contiguous in row-major, so a batch of consecutive rows is a contiguous
// slice: batch() is one async device→device memcpy per matrix. shuffle() permutes the
// rows on-device (rather than gathering scattered indices per batch) precisely so that
// batches remain contiguous. Nothing allocates per step.

class Dataset {
public:
    Dataset(Matrix X, Matrix Y);   // takes ownership
    ~Dataset();

    Dataset(const Dataset&)            = delete;
    Dataset& operator=(const Dataset&) = delete;

    i32 size()     const { return X_.rows(); }
    i32 features() const { return X_.cols(); }
    i32 classes()  const { return Y_.cols(); }

    // Whole batches only — the engine's buffers are fixed-size, so a short trailing
    // batch is dropped.
    i32 num_batches(i32 batch_size) const { return batch_size > 0 ? size() / batch_size : 0; }

    // Copy rows [i*B, (i+1)*B) into the caller's preallocated Xout(B×features),
    // Yout(B×classes). Async on g_compute_stream; no allocation, no sync.
    void batch(i32 i, Matrix& Xout, Matrix& Yout) const;

    // Permute rows on-device with a single random permutation applied to BOTH X and Y,
    // so every image keeps its own label. Uses mlkit's module RNG — mlkit_seed() makes
    // shuffling reproducible.
    void shuffle();

    const Matrix& X() const { return X_; }
    const Matrix& Y() const { return Y_; }

private:
    Matrix X_, Y_;
    Matrix Xs_, Ys_;         // gather destination (swapped in after each shuffle)
    i32*   d_perm_ = nullptr; // device row permutation (Matrix is f32-only), allocated once
};
