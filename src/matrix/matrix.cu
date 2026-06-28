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
