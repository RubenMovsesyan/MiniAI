#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <io/io.cuh>
#include <harness_test.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Real MNIST IDX files. Override the directory with MNIST_DIR; defaults to
// $HOME/Downloads/ml_training. If they aren't there the suite reports a skip
// rather than failing (the files are large and not in the repo).
static std::string data_dir() {
    if (const char* d = std::getenv("MNIST_DIR")) return d;
    const char* home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/Downloads/ml_training";
}
static std::string path_of(const char* name) { return data_dir() + "/" + name; }

static bool have_data() {
    FILE* f = std::fopen(path_of("train-images.idx3-ubyte").c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

static std::vector<f32> download(const Matrix& m) {
    usize n = (usize)m.rows() * m.cols();
    std::vector<f32> h(n);
    cudaMemcpy(h.data(), m.data, n * sizeof(f32), cudaMemcpyDeviceToHost);
    return h;
}

// ─── Header / dims parsing (proves the big-endian u32 header swap works) ────────

static void test_train_images_header() {
    IdxTensor t = load_idx(path_of("train-images.idx3-ubyte").c_str());
    bool ok = t.ok()
           && t.dtype == IdxType::U8
           && t.dims.size() == 3
           && t.dims[0] == 60000 && t.dims[1] == 28 && t.dims[2] == 28
           && t.count() == 60000ull * 28 * 28
           && t.raw.size() == t.count();
    if (!ok && t.ok())
        RLOG(LL_ERROR, "train images: dtype 0x%02X, ndims %zu", (u32)t.dtype, t.dims.size());
    record(ok, "idx_train_images_header");
}

static void test_train_labels_values() {
    IdxTensor t = load_idx(path_of("train-labels.idx1-ubyte").c_str());
    // Canonical MNIST training labels start 5, 0, 4, 1, 9, 2, 1, 3
    const u8 expected[8] = {5, 0, 4, 1, 9, 2, 1, 3};
    bool ok = t.ok() && t.dtype == IdxType::U8
           && t.dims.size() == 1 && t.dims[0] == 60000;
    if (ok)
        for (i32 i = 0; i < 8; i++)
            if (t.raw[i] != expected[i]) {
                RLOG(LL_ERROR, "train label[%d] = %u, expected %u", i, t.raw[i], expected[i]);
                ok = false;
            }
    record(ok, "idx_train_labels_values");
}

static void test_t10k_labels_values() {
    IdxTensor t = load_idx(path_of("t10k-labels.idx1-ubyte").c_str());
    // Canonical MNIST test labels start 7, 2, 1, 0, 4, 1, 4, 9, 5, 9
    const u8 expected[10] = {7, 2, 1, 0, 4, 1, 4, 9, 5, 9};
    bool ok = t.ok() && t.dims.size() == 1 && t.dims[0] == 10000;
    if (ok)
        for (i32 i = 0; i < 10; i++)
            if (t.raw[i] != expected[i]) {
                RLOG(LL_ERROR, "t10k label[%d] = %u, expected %u", i, t.raw[i], expected[i]);
                ok = false;
            }
    record(ok, "idx_t10k_labels_values");
}

// ─── Dataset build: flatten, /255 normalize, one-hot ────────────────────────────

static void test_dataset_shape_and_normalization() {
    IdxTensor imgs = load_idx(path_of("t10k-images.idx3-ubyte").c_str());
    IdxDataset d = load_idx_dataset(path_of("t10k-images.idx3-ubyte").c_str(),
                                    path_of("t10k-labels.idx1-ubyte").c_str(), 10);
    bool ok = d.ok() && d.n == 10000 && d.features == 784 && d.classes == 10
           && d.X.rows() == 10000 && d.X.cols() == 784;
    if (!ok) { record(false, "idx_dataset_shape"); return; }

    auto X = download(d.X);

    // Every pixel in [0,1], and X[i] == raw[i]/255 exactly (row-major flatten).
    bool in_range = true, matches_raw = true, any_nonzero = false;
    for (usize i = 0; i < X.size(); i++) {
        if (X[i] < 0.0f || X[i] > 1.0f) { in_range = false; break; }
        if (X[i] > 0.0f) any_nonzero = true;
    }
    for (usize i = 0; i < 5000; i++) {   // spot-check the flatten against the raw bytes
        f32 want = (f32)imgs.raw[i] / 255.0f;
        if (fabsf(X[i] - want) > 1e-6f) { matches_raw = false; break; }
    }
    record(in_range && matches_raw && any_nonzero, "idx_dataset_shape_normalization");
}

static void test_dataset_one_hot() {
    IdxDataset d = load_idx_dataset(path_of("t10k-images.idx3-ubyte").c_str(),
                                    path_of("t10k-labels.idx1-ubyte").c_str(), 10);
    if (!d.ok()) { record(false, "idx_dataset_one_hot"); return; }

    auto Y = download(d.Y);
    const i32 expected[10] = {7, 2, 1, 0, 4, 1, 4, 9, 5, 9};

    bool ok = true;
    // First 10 rows: argmax must equal the known labels.
    for (i32 r = 0; r < 10 && ok; r++) {
        i32 argmax = 0;
        for (i32 c = 1; c < 10; c++)
            if (Y[(usize)r * 10 + c] > Y[(usize)r * 10 + argmax]) argmax = c;
        if (argmax != expected[r]) {
            RLOG(LL_ERROR, "one-hot row %d argmax %d, expected %d", r, argmax, expected[r]);
            ok = false;
        }
    }
    // Every row is a valid one-hot: sums to exactly 1.
    for (i32 r = 0; r < d.n && ok; r++) {
        f32 s = 0.0f;
        for (i32 c = 0; c < 10; c++) s += Y[(usize)r * 10 + c];
        if (fabsf(s - 1.0f) > 1e-6f) {
            RLOG(LL_ERROR, "one-hot row %d sums to %f", r, s);
            ok = false;
        }
    }
    record(ok, "idx_dataset_one_hot");
}

// ─── Malformed input: report, don't crash ───────────────────────────────────────

static void test_malformed() {
    RLOG(LL_INFO, "(the two errors below are expected — negative tests)");
    IdxTensor missing = load_idx(path_of("definitely-not-here.idx3-ubyte").c_str());
    IdxTensor not_idx = load_idx("CLAUDE.md");   // real file, bad magic
    record(!missing.ok() && !not_idx.ok(), "idx_malformed_rejected");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    initLog(65536);

    if (!have_data()) {
        RLOG(LL_WARN, "MNIST IDX files not found in %s — skipping io tests "
                      "(set MNIST_DIR to override)", data_dir().c_str());
        return 0;
    }

    test_train_images_header();
    test_train_labels_values();
    test_t10k_labels_values();
    test_dataset_shape_and_normalization();
    test_dataset_one_hot();
    test_malformed();
    return testSummary();
}
