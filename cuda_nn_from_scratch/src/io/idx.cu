#include <rlog.h>
#include <io/idx.cuh>

#include <fstream>

usize idx_elem_size(IdxType t) {
    switch (t) {
        case IdxType::U8:
        case IdxType::I8:  return 1;
        case IdxType::I16: return 2;
        case IdxType::I32:
        case IdxType::F32: return 4;
        case IdxType::F64: return 8;
    }
    return 0;
}

usize IdxTensor::count() const {
    if (dims.empty()) return 0;
    usize n = 1;
    for (i32 d : dims) n *= (usize)d;
    return n;
}

static u32 swap_endian(u32 val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

// Reverse the bytes of each element in place (IDX stores multi-byte types big-endian).
static void swap_elements(std::vector<u8>& raw, usize elem_size) {
    if (elem_size < 2) return;
    for (usize i = 0; i + elem_size <= raw.size(); i += elem_size)
        for (usize a = 0, b = elem_size - 1; a < b; a++, b--)
            std::swap(raw[i + a], raw[i + b]);
}

static bool known_dtype(u8 t) {
    switch (t) {
        case 0x08: case 0x09: case 0x0B: case 0x0C: case 0x0D: case 0x0E: return true;
        default: return false;
    }
}

IdxTensor load_idx(const char* path) {
    IdxTensor t;

    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) {
        RLOG(LL_ERROR, "load_idx: cannot open %s", path);
        return t;
    }

    u8 magic[4];
    f.read(reinterpret_cast<char*>(magic), 4);
    if (!f || magic[0] != 0x00 || magic[1] != 0x00) {
        RLOG(LL_ERROR, "load_idx: bad magic in %s (expected 0x00 0x00 prefix)", path);
        return t;
    }
    if (!known_dtype(magic[2])) {
        RLOG(LL_ERROR, "load_idx: unknown dtype 0x%02X in %s", magic[2], path);
        return t;
    }
    IdxType dtype = (IdxType)magic[2];
    i32     ndims = (i32)magic[3];
    if (ndims < 1) {
        RLOG(LL_ERROR, "load_idx: bad ndims %d in %s", ndims, path);
        return t;
    }

    std::vector<i32> dims(ndims);
    for (i32 i = 0; i < ndims; i++) {
        u32 d;
        f.read(reinterpret_cast<char*>(&d), 4);
        if (!f) {
            RLOG(LL_ERROR, "load_idx: truncated header in %s", path);
            return t;
        }
        dims[i] = (i32)swap_endian(d);
    }

    t.dtype = dtype;
    t.dims  = dims;

    usize elem  = idx_elem_size(dtype);
    usize bytes = t.count() * elem;
    t.raw.resize(bytes);
    f.read(reinterpret_cast<char*>(t.raw.data()), (std::streamsize)bytes);
    if ((usize)f.gcount() != bytes) {
        RLOG(LL_ERROR, "load_idx: truncated data in %s (got %lld of %zu bytes)",
             path, (long long)f.gcount(), bytes);
        t.dims.clear();   // ok() == false
        t.raw.clear();
        return t;
    }

    swap_elements(t.raw, elem);
    return t;
}

IdxDataset load_idx_dataset(const char* image_path, const char* label_path, i32 num_classes) {
    IdxTensor images = load_idx(image_path);
    IdxTensor labels = load_idx(label_path);
    if (!images.ok() || !labels.ok()) return IdxDataset(0, 0, 0);

    if (images.dtype != IdxType::U8 || images.dims.size() != 3) {
        RLOG(LL_ERROR, "load_idx_dataset: expected a 3-D u8 image tensor in %s", image_path);
        return IdxDataset(0, 0, 0);
    }
    if (labels.dtype != IdxType::U8 || labels.dims.size() != 1) {
        RLOG(LL_ERROR, "load_idx_dataset: expected a 1-D u8 label tensor in %s", label_path);
        return IdxDataset(0, 0, 0);
    }
    if (images.dims[0] != labels.dims[0]) {
        RLOG(LL_ERROR, "load_idx_dataset: image count %d != label count %d",
             images.dims[0], labels.dims[0]);
        return IdxDataset(0, 0, 0);
    }

    i32 n        = images.dims[0];
    i32 features = images.dims[1] * images.dims[2];   // flatten each image to one row

    // X: pixels /255 → [0,1]. Y: one-hot. Row-major, batch = rows.
    std::vector<f32> hX((usize)n * features);
    for (usize i = 0; i < hX.size(); i++) hX[i] = (f32)images.raw[i] / 255.0f;

    std::vector<f32> hY((usize)n * num_classes, 0.0f);
    for (i32 i = 0; i < n; i++) {
        i32 label = (i32)labels.raw[i];
        if (label < 0 || label >= num_classes) {
            RLOG(LL_ERROR, "load_idx_dataset: label %d out of range [0,%d)", label, num_classes);
            return IdxDataset(0, 0, 0);
        }
        hY[(usize)i * num_classes + label] = 1.0f;
    }

    IdxDataset d(n, features, num_classes);
    cudaMemcpy(d.X.data, hX.data(), hX.size() * sizeof(f32), cudaMemcpyHostToDevice);
    cudaMemcpy(d.Y.data, hY.data(), hY.size() * sizeof(f32), cudaMemcpyHostToDevice);
    RLOG(LL_INFO, "loaded %d samples: %d features, %d classes", n, features, num_classes);
    return d;
}
