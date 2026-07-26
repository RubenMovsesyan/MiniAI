#pragma once

#include <common.h>
#include <matrix/matrix.cuh>
#include <harness_csv.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

inline const char* DATA_ROOT = "src/matrix/tests/data";

// ─── Path helpers ─────────────────────────────────────────────────────────────
#define PA(buf, name)     snprintf(buf, sizeof(buf), "%s/inputs/%s/A.csv",   DATA_ROOT, name)
#define PB(buf, name)     snprintf(buf, sizeof(buf), "%s/inputs/%s/B.csv",   DATA_ROOT, name)
#define PCOL(buf, name)   snprintf(buf, sizeof(buf), "%s/inputs/%s/col.csv", DATA_ROOT, name)
#define PROW(buf, name)   snprintf(buf, sizeof(buf), "%s/inputs/%s/row.csv", DATA_ROOT, name)
#define PE(buf, op, name) snprintf(buf, sizeof(buf), "%s/expected/%s/%s.csv", DATA_ROOT, op, name)
#define LBL(buf, op, name) snprintf(buf, sizeof(buf), "%s/%s", op, name)

// ─── CSV ↔ Matrix glue ────────────────────────────────────────────────────────

inline Matrix matLoad(const char* path) {
    i32 rows, cols;
    f32* h = csvLoad(path, &rows, &cols);
    Matrix m(rows, cols);
    cudaMemcpy(m.data, h, (usize)rows * cols * sizeof(f32), cudaMemcpyHostToDevice);
    free(h);
    return m;
}

// Compare a GPU Matrix against an expected CSV within a relative tolerance.
inline bool matCheckCSV(const Matrix& result, const char* expected_path, f32 epsilon = 1e-4f) {
    i32 exp_rows, exp_cols;
    f32* h_exp = csvLoad(expected_path, &exp_rows, &exp_cols);
    if (!h_exp) return false;
    if (exp_rows != result.rows() || exp_cols != result.cols()) {
        RLOG(LL_ERROR, "shape mismatch: result %dx%d vs expected %dx%d",
             result.rows(), result.cols(), exp_rows, exp_cols);
        free(h_exp); return false;
    }
    i32 n = exp_rows * exp_cols;
    f32* h_res = (f32*)malloc((usize)n * sizeof(f32));
    cudaMemcpy(h_res, result.data, (usize)n * sizeof(f32), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (i32 i = 0; i < n && ok; i++) {
        if (fabsf(h_res[i] - h_exp[i]) > epsilon * fmaxf(1.0f, fabsf(h_exp[i]))) {
            RLOG(LL_ERROR, "mismatch at element %d: got %.6f expected %.6f",
                 i, (double)h_res[i], (double)h_exp[i]);
            ok = false;
        }
    }
    free(h_res); free(h_exp);
    return ok;
}

// ─── Variant lists ────────────────────────────────────────────────────────────

// All size/type combinations
inline const char* ALL_VARIANTS[] = {
    "3x3_rand_f32",    "3x3_struct_f32",    "3x3_rand_i32",    "3x3_struct_i32",
    "3x5_rand_f32",    "3x5_struct_f32",    "3x5_rand_i32",    "3x5_struct_i32",
    "4x4_rand_f32",    "4x4_struct_f32",    "4x4_rand_i32",    "4x4_struct_i32",
    "4x7_rand_f32",    "4x7_struct_f32",    "4x7_rand_i32",    "4x7_struct_i32",
    "8x8_rand_f32",    "8x8_struct_f32",    "8x8_rand_i32",    "8x8_struct_i32",
    "16x16_rand_f32",  "16x16_struct_f32",  "16x16_rand_i32",  "16x16_struct_i32",
    "16x32_rand_f32",  "16x32_struct_f32",  "16x32_rand_i32",  "16x32_struct_i32",
    "64x64_rand_f32",  "64x64_struct_f32",  "64x64_rand_i32",  "64x64_struct_i32",
    "128x128_rand_f32","128x128_struct_f32","128x128_rand_i32","128x128_struct_i32",
    "128x256_rand_f32","128x256_struct_f32","128x256_rand_i32","128x256_struct_i32",
    "512x512_rand_f32",
    "1024x1024_rand_f32",
    nullptr,
};

// Square sizes only — used for matmul and matmul-chain tests
inline const char* SQUARE_VARIANTS[] = {
    "3x3_rand_f32",    "3x3_struct_f32",    "3x3_rand_i32",    "3x3_struct_i32",
    "4x4_rand_f32",    "4x4_struct_f32",    "4x4_rand_i32",    "4x4_struct_i32",
    "8x8_rand_f32",    "8x8_struct_f32",    "8x8_rand_i32",    "8x8_struct_i32",
    "16x16_rand_f32",  "16x16_struct_f32",  "16x16_rand_i32",  "16x16_struct_i32",
    "64x64_rand_f32",  "64x64_struct_f32",  "64x64_rand_i32",  "64x64_struct_i32",
    "128x128_rand_f32","128x128_struct_f32","128x128_rand_i32","128x128_struct_i32",
    "512x512_rand_f32",
    "1024x1024_rand_f32",
    nullptr,
};
