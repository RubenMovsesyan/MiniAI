#define RLOG_IMPLEMENTATION
#include <matrix/matrix.cuh>

Matrix::Matrix(i32 rows, i32 cols) : _rows(rows), _cols(cols), data(nullptr) {
  cudaMalloc(&data, rows * cols * sizeof(f32));
}

Matrix::~Matrix() {
  if (data)
    cudaFree(data);
}

Matrix::Matrix(Matrix &&other) noexcept
    : data(other.data), _rows(other._rows), _cols(other._cols) {
  other.data = nullptr;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
  if (this != &other) {
    if (data)
      cudaFree(data);
    data = other.data;
    _rows = other._rows;
    _cols = other._cols;
    other.data = nullptr;
  }
  return *this;
}

Matrix Matrix::eval() const {
  Matrix out(_rows, _cols);
  cudaMemcpy(out.data, data, _rows * _cols * sizeof(f32),
             cudaMemcpyDeviceToDevice);
  return out;
}

constexpr i32 TILE = 16; // block is TILE×TILE (256 threads)

// Shared-memory tiled GEMM. One thread per output element. Wins on small matrices
// (<= 128) but is shared-memory-bound on larger L2-resident sizes — see gemmKernel2D.
__global__ void gemmKernelTiled(const f32 *A, const f32 *B, f32 *C, i32 M, i32 K,
                                i32 N) {
  // ponytail: +1 column pads each row to a different bank alignment, avoiding
  // shared-memory bank conflicts. As[ty][k] is tx-independent (broadcast read);
  // Bs[k][tx] is column-contiguous.
  __shared__ f32 As[TILE][TILE + 1];
  __shared__ f32 Bs[TILE][TILE + 1];

  i32 ty = threadIdx.y, tx = threadIdx.x;
  i32 row = blockIdx.y * TILE + ty; // output row this thread owns
  i32 col = blockIdx.x * TILE + tx; // output col this thread owns

  f32 acc = 0.0f;
  i32 nTiles = (K + TILE - 1) / TILE;
  for (i32 t = 0; t < nTiles; t++) {
    i32 aCol = t * TILE + tx; // column of A this thread loads
    i32 bRow = t * TILE + ty; // row of B this thread loads
    // cooperative load; zero-fill out-of-range so partial edge tiles stay
    // correct
    As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads(); // tiles fully populated before use

    for (i32 k = 0; k < TILE; k++)
      acc += As[ty][k] * Bs[k][tx];
    __syncthreads(); // finish reading before next load overwrites
  }

  if (row < M && col < N)
    C[row * N + col] = acc;
}

// 2D block-tiling GEMM (siboehm "kernel 5"). Each thread computes a TM×TN register
// tile via an outer product, so the BK inner step does TM+TN shared loads for TM*TN
// MACs (~4 MACs/load at 8×8). Raises arithmetic intensity off the shared-memory
// bottleneck → wins on the larger L2-resident sizes (512+).
constexpr i32 BM = 64, BN = 64, BK = 8, TM = 8, TN = 8;
constexpr i32 NUM_THREADS = (BM * BN) / (TM * TN); // 64

__global__ __launch_bounds__(NUM_THREADS) void
gemmKernel2D(const f32 *A, const f32 *B, f32 *C, i32 M, i32 K, i32 N) {
  __shared__ f32 As[BK * BM]; // stored transposed: As[k*BM + r] → contiguous regM loads
  __shared__ f32 Bs[BK * BN];

  i32 blockRow = blockIdx.y, blockCol = blockIdx.x;
  // this thread's position inside the BM×BN output tile
  i32 threadCol = threadIdx.x % (BN / TN); // 0..7
  i32 threadRow = threadIdx.x / (BN / TN); // 0..7

  f32 acc[TM][TN] = {}; // register accumulators
  f32 regM[TM], regN[TN];

  i32 nTiles = (K + BK - 1) / BK;
  for (i32 t = 0; t < nTiles; t++) {
    // cooperative load (64 threads, BM*BK=512 elems → 8 strided iters each)
    for (i32 li = threadIdx.x; li < BM * BK; li += NUM_THREADS) {
      i32 r = li / BK, c = li % BK;
      i32 gRow = blockRow * BM + r, gCol = t * BK + c;
      As[c * BM + r] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
    }
    for (i32 li = threadIdx.x; li < BK * BN; li += NUM_THREADS) {
      i32 r = li / BN, c = li % BN;
      i32 gRow = t * BK + r, gCol = blockCol * BN + c;
      Bs[r * BN + c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
    }
    __syncthreads();

    // register outer-product accumulation
    for (i32 k = 0; k < BK; k++) {
      for (i32 i = 0; i < TM; i++)
        regM[i] = As[k * BM + threadRow * TM + i];
      for (i32 j = 0; j < TN; j++)
        regN[j] = Bs[k * BN + threadCol * TN + j];
      for (i32 i = 0; i < TM; i++)
        for (i32 j = 0; j < TN; j++)
          acc[i][j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

  // write the TM×TN register tile (bounds-guarded for edge blocks)
  for (i32 i = 0; i < TM; i++) {
    i32 gRow = blockRow * BM + threadRow * TM + i;
    for (i32 j = 0; j < TN; j++) {
      i32 gCol = blockCol * BN + threadCol * TN + j;
      if (gRow < M && gCol < N)
        C[gRow * N + gCol] = acc[i][j];
    }
  }
}

void matmulDispatch(const f32 *A, const f32 *B, f32 *C, i32 M, i32 K, i32 N) {
  // ponytail: 2D tiling wins large L2-resident sizes; the tiled kernel wins small.
  // THRESH is the bench-tunable crossover (square variants jump 128 → 512).
  constexpr i32 THRESH = 256;
  if (M >= THRESH && N >= THRESH) {
    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemmKernel2D<<<grid, block>>>(A, B, C, M, K, N);
  } else {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemmKernelTiled<<<grid, block>>>(A, B, C, M, K, N);
  }
  cudaDeviceSynchronize();
}
