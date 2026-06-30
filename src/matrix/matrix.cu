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

__global__ void gemmKernel(const f32 *A, const f32 *B, f32 *C, i32 M, i32 K,
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

void matmulDispatch(const f32 *A, const f32 *B, f32 *C, i32 M, i32 K, i32 N) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemmKernel<<<grid, block>>>(A, B, C, M, K, N);
  cudaDeviceSynchronize();
}
