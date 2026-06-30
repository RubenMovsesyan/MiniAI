#define RLOG_IMPLEMENTATION
#include <matrix/matrix.cuh>

Matrix::Matrix(i32 rows, i32 cols) : _rows(rows), _cols(cols), data(nullptr) {
    cudaMalloc(&data, rows * cols * sizeof(f32));
}

Matrix::~Matrix() {
    if (data) cudaFree(data);
}

Matrix::Matrix(Matrix&& other) noexcept
    : data(other.data), _rows(other._rows), _cols(other._cols) {
    other.data = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        if (data) cudaFree(data);
        data   = other.data;
        _rows  = other._rows;
        _cols  = other._cols;
        other.data = nullptr;
    }
    return *this;
}

Matrix Matrix::eval() const {
    Matrix out(_rows, _cols);
    cudaMemcpy(out.data, data, _rows * _cols * sizeof(f32), cudaMemcpyDeviceToDevice);
    return out;
}

__global__ void gemmKernel(const f32* A, const f32* B, f32* C, i32 M, i32 K, i32 N) {
    i32 row = blockIdx.y * blockDim.y + threadIdx.y;
    i32 col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    f32 acc = 0.0f;
    for (i32 k = 0; k < K; k++)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

void matmulDispatch(const f32* A, const f32* B, f32* C, i32 M, i32 K, i32 N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemmKernel<<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
