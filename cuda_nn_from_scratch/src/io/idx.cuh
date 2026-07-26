#pragma once

#include <common.h>
#include <matrix/matrix.cuh>

#include <vector>

// ─── IDX file format ────────────────────────────────────────────────────────────
// Header: 4-byte magic = { 0x00, 0x00, dtype, ndims }, then `ndims` big-endian u32
// dimensions, then the raw element data (also big-endian for multi-byte types).
//   MNIST images: magic 0x00000803 → dtype 0x08 (u8), 3 dims {N, rows, cols}
//   MNIST labels: magic 0x00000801 → dtype 0x08 (u8), 1 dim  {N}
// Parsing the magic properly (rather than hardcoding 2051/2049) makes this work for
// any IDX file — MNIST, Fashion-MNIST, EMNIST, and non-u8 dtypes.

enum class IdxType : u8 {
    U8  = 0x08,
    I8  = 0x09,
    I16 = 0x0B,
    I32 = 0x0C,
    F32 = 0x0D,
    F64 = 0x0E,
};

usize idx_elem_size(IdxType t);

struct IdxTensor {
    IdxType          dtype = IdxType::U8;
    std::vector<i32> dims;   // {60000, 28, 28} or {60000}
    std::vector<u8>  raw;    // element bytes; multi-byte types already byte-swapped

    usize count() const;     // product of dims (0 if dims is empty)
    bool  ok() const { return !dims.empty(); }
};

// Parse an IDX file. On failure: RLOG_ERROR + a tensor with empty dims (ok() == false).
IdxTensor load_idx(const char* path);

// ─── Dataset builder ────────────────────────────────────────────────────────────
// Converts an IDX image + label file pair into GPU matrices in the engine's layout
// (batch = rows):
//   X : (N × features)  f32, each image flattened to one row, pixels scaled /255 → [0,1]
//   Y : (N × classes)   f32, one-hot labels
// Host-side: builds the f32 arrays then does one cudaMemcpy each. Runs once at startup.

struct IdxDataset {
    Matrix X;
    Matrix Y;
    i32 n, features, classes;

    // Matrix has no default ctor, so the buffers are sized here. A failed load returns
    // IdxDataset(0,0,0) — dummy 1×1 buffers with ok() == false.
    IdxDataset(i32 n_, i32 features_, i32 classes_)
        : X(n_ > 0 ? n_ : 1, features_ > 0 ? features_ : 1),
          Y(n_ > 0 ? n_ : 1, classes_  > 0 ? classes_  : 1),
          n(n_), features(features_), classes(classes_) {}

    bool ok() const { return n > 0; }
};

IdxDataset load_idx_dataset(const char* image_path, const char* label_path,
                            i32 num_classes = 10);
