#include <rlog.h>
#include <agg/aggregations.cuh>

// ─── Row sum kernel ───────────────────────────────────────────────────────
// Each block reduces one row. Uses shared memory + syncthreads for block reduction.

__global__ static void rowSumKernel(const f32* x, f32* out, i32 rows, i32 cols) {
    extern __shared__ f32 sdata[];

    i32 row = blockIdx.x;
    if (row >= rows) return;

    f32 sum = 0.0f;
    for (i32 col = threadIdx.x; col < cols; col += blockDim.x)
        sum += x[row * cols + col];

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (i32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[row] = sdata[0];
}

Matrix row_sum(const Matrix& x) {
    Matrix out(x.rows(), 1);
    rowSumKernel<<<x.rows(), 256, 256 * sizeof(f32), g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
    return out;
}

void row_sum(const Matrix& x, Matrix& out) {
    rowSumKernel<<<x.rows(), 256, 256 * sizeof(f32), g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
}

// ─── Column sum kernel ────────────────────────────────────────────────────
// Each block reduces one column. Uses shared memory + syncthreads for block reduction.

__global__ static void colSumKernel(const f32* x, f32* out, i32 rows, i32 cols) {
    extern __shared__ f32 sdata[];

    i32 col = blockIdx.x;
    if (col >= cols) return;

    f32 sum = 0.0f;
    for (i32 row = threadIdx.x; row < rows; row += blockDim.x)
        sum += x[row * cols + col];

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (i32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[col] = sdata[0];
}

Matrix col_sum(const Matrix& x) {
    Matrix out(1, x.cols());
    colSumKernel<<<x.cols(), 256, 256 * sizeof(f32), g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
    return out;
}

void col_sum(const Matrix& x, Matrix& out) {
    colSumKernel<<<x.cols(), 256, 256 * sizeof(f32), g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
}

// ─── Total sum (via chaining) ─────────────────────────────────────────────

Matrix sum(const Matrix& x) {
    Matrix row_sums = row_sum(x);
    Matrix result = col_sum(row_sums);
    return result;
}

void sum(const Matrix& x, Matrix& out) {
    Matrix row_sums = row_sum(x);
    col_sum(row_sums, out);
}

// ─── Row max kernel ───────────────────────────────────────────────────────
// Each block reduces one row. Uses shared memory + syncthreads for block reduction.

__global__ static void rowMaxKernel(const f32* x, f32* out, i32 rows, i32 cols) {
    extern __shared__ f32 sdata[];

    i32 row = blockIdx.x;
    if (row >= rows) return;

    f32 maxval = -INFINITY;
    for (i32 col = threadIdx.x; col < cols; col += blockDim.x)
        maxval = fmaxf(maxval, x[row * cols + col]);

    sdata[threadIdx.x] = maxval;
    __syncthreads();

    for (i32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[row] = sdata[0];
}

Matrix row_max(const Matrix& x) {
    Matrix out(x.rows(), 1);
    rowMaxKernel<<<x.rows(), 256, 256 * sizeof(f32), g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
    return out;
}

void row_max(const Matrix& x, Matrix& out) {
    rowMaxKernel<<<x.rows(), 256, 256 * sizeof(f32), g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
}

// ─── Column max kernel ────────────────────────────────────────────────────
// Each block reduces one column. Uses shared memory + syncthreads for block reduction.

__global__ static void colMaxKernel(const f32* x, f32* out, i32 rows, i32 cols) {
    extern __shared__ f32 sdata[];

    i32 col = blockIdx.x;
    if (col >= cols) return;

    f32 maxval = -INFINITY;
    for (i32 row = threadIdx.x; row < rows; row += blockDim.x)
        maxval = fmaxf(maxval, x[row * cols + col]);

    sdata[threadIdx.x] = maxval;
    __syncthreads();

    for (i32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[col] = sdata[0];
}

Matrix col_max(const Matrix& x) {
    Matrix out(1, x.cols());
    colMaxKernel<<<x.cols(), 256, 256 * sizeof(f32), g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
    return out;
}

void col_max(const Matrix& x, Matrix& out) {
    colMaxKernel<<<x.cols(), 256, 256 * sizeof(f32), g_compute_stream>>>(x.data, out.data, x.rows(), x.cols());
}

// ─── Total max (via chaining) ─────────────────────────────────────────────

Matrix max(const Matrix& x) {
    Matrix row_maxs = row_max(x);
    Matrix result = col_max(row_maxs);
    return result;
}

void max(const Matrix& x, Matrix& out) {
    Matrix row_maxs = row_max(x);
    col_max(row_maxs, out);
}

// ─── Row argmax kernel ────────────────────────────────────────────────────
// Same block-per-row shared-memory reduction as rowMaxKernel, but carries the index
// alongside the value. Shared memory holds two arrays: values then indices.
// Ties resolve to the lowest index (standard argmax semantics).

__global__ static void rowArgmaxKernel(const f32* x, f32* out, i32 rows, i32 cols) {
    extern __shared__ f32 sdata[];
    f32* sval = sdata;
    f32* sidx = sdata + blockDim.x;

    i32 row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;

    // Threads with no elements keep -INFINITY, so they never win the reduction.
    f32 best = -INFINITY;
    i32 best_idx = 0;
    for (i32 c = tid; c < cols; c += blockDim.x) {
        f32 v = x[row * cols + c];
        if (v > best) { best = v; best_idx = c; }
    }

    sval[tid] = best;
    sidx[tid] = (f32)best_idx;
    __syncthreads();

    for (i32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            f32 ov = sval[tid + s];
            f32 oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) out[row] = sidx[0];
}

Matrix row_argmax(const Matrix& x) {
    Matrix out(x.rows(), 1);
    row_argmax(x, out);
    return out;
}

void row_argmax(const Matrix& x, Matrix& out) {
    rowArgmaxKernel<<<x.rows(), 256, 2 * 256 * sizeof(f32), g_compute_stream>>>(
        x.data, out.data, x.rows(), x.cols());
}
