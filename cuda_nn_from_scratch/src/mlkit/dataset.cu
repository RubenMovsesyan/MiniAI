#include <rlog.h>
#include <mlkit/dataset.cuh>
#include <mlkit/init.cuh>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

// out[r, c] = in[perm[r], c] — permute rows. The SAME perm is applied to X and Y, so
// every sample keeps its own label.
__global__ static void gatherRowsKernel(const f32* in, f32* out, const i32* perm,
                                        i32 rows, i32 cols) {
    i32 r = blockIdx.y * blockDim.y + threadIdx.y;
    i32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols)
        out[r * cols + c] = in[perm[r] * cols + c];
}

Dataset::Dataset(Matrix X, Matrix Y)
    : X_(std::move(X)), Y_(std::move(Y)),
      Xs_(X_.rows(), X_.cols()), Ys_(Y_.rows(), Y_.cols()) {
    cudaMalloc(&d_perm_, (usize)X_.rows() * sizeof(i32));
}

Dataset::~Dataset() {
    if (d_perm_) cudaFree(d_perm_);
}

void Dataset::batch(i32 i, Matrix& Xout, Matrix& Yout) const {
    i32 B = Xout.rows();
    if (i < 0 || (i + 1) * B > size()) {
        RLOG(LL_ERROR, "Dataset::batch: batch %d of size %d out of range (N=%d)", i, B, size());
        return;
    }
    // Rows are contiguous → one async D2D copy each. No allocation, no sync.
    usize xoff = (usize)i * B * features();
    usize yoff = (usize)i * B * classes();
    cudaMemcpyAsync(Xout.data, X_.data + xoff, (usize)B * features() * sizeof(f32),
                    cudaMemcpyDeviceToDevice, g_compute_stream);
    cudaMemcpyAsync(Yout.data, Y_.data + yoff, (usize)B * classes() * sizeof(f32),
                    cudaMemcpyDeviceToDevice, g_compute_stream);
}

void Dataset::shuffle() {
    i32 n = size();

    std::vector<i32> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), mlkit_rng());   // reproducible via mlkit_seed

    // Synchronous H2D: perm is pageable host memory that dies at the end of this call.
    // shuffle() runs once per epoch, so the stall is irrelevant.
    cudaMemcpy(d_perm_, perm.data(), (usize)n * sizeof(i32), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 gridX((features() + 15) / 16, (n + 15) / 16);
    dim3 gridY((classes()  + 15) / 16, (n + 15) / 16);
    gatherRowsKernel<<<gridX, block, 0, g_compute_stream>>>(X_.data, Xs_.data, d_perm_, n, features());
    gatherRowsKernel<<<gridY, block, 0, g_compute_stream>>>(Y_.data, Ys_.data, d_perm_, n, classes());

    // Swap the buffers in (host-side pointer swap; the gather kernels are ordered ahead
    // of any later read on the same stream).
    std::swap(X_, Xs_);
    std::swap(Y_, Ys_);
}
