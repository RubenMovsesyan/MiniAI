#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <mlkit/mlkit.cuh>
#include <io/io.cuh>

#include <chrono>
#include <cstdlib>
#include <string>

// ─── Training parameters — edit these, then `./build && .build/mnist` ───────────

struct TrainConfig {
    // Where the IDX files live. nullptr → $MNIST_DIR, else $HOME/Downloads/ml_training.
    const char* data_dir      = nullptr;

    i32 hidden                = 128;    // hidden layer width
    i32 epochs                = 27;
    i32 batch_size            = 100;    // divides 60000 AND 10000 → nothing is dropped
    f32 lr                    = 0.1f;   // SGD learning rate
    u32 seed                  = 42;     // reproducible weights + shuffling
    i32 eval_interval         = 100;    // train steps between loss readbacks (0 = never)
};

static TrainConfig cfg;   // ← tweak the values above

// ────────────────────────────────────────────────────────────────────────────────

static std::string data_dir() {
    if (cfg.data_dir) return cfg.data_dir;
    if (const char* d = std::getenv("MNIST_DIR")) return d;
    const char* home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/Downloads/ml_training";
}

// Run the network over every whole batch of `ds` and return the accuracy.
// The engine's buffers are fixed-size, so a trailing partial batch is skipped —
// value() reports over meter.total(), which is logged, so the number is never
// silently computed over fewer samples than you think.
static f32 evaluate(Network& net, const Dataset& ds, AccuracyMeter& meter,
                    Matrix& Xb, Matrix& Yb, i32 batch_size) {
    meter.reset();
    i32 nb = ds.num_batches(batch_size);
    for (i32 i = 0; i < nb; i++) {
        ds.batch(i, Xb, Yb);
        meter.update(net.forward(Xb), Yb);
    }
    return meter.value();
}

int main() {
    initLog(65536);

    const std::string dir = data_dir();
    IdxDataset raw_train = load_idx_dataset((dir + "/train-images.idx3-ubyte").c_str(),
                                            (dir + "/train-labels.idx1-ubyte").c_str(), 10);
    IdxDataset raw_test  = load_idx_dataset((dir + "/t10k-images.idx3-ubyte").c_str(),
                                            (dir + "/t10k-labels.idx1-ubyte").c_str(), 10);
    if (!raw_train.ok() || !raw_test.ok()) {
        RLOG(LL_ERROR, "could not load MNIST from %s "
                       "(set MNIST_DIR or TrainConfig::data_dir)", dir.c_str());
        return 1;
    }

    Dataset train(std::move(raw_train.X), std::move(raw_train.Y));
    Dataset test (std::move(raw_test.X),  std::move(raw_test.Y));

    const i32 B = cfg.batch_size;

    mlkit_seed(cfg.seed);
    Network net = NetworkBuilder(B, train.features())
        .dense(cfg.hidden,       Activation::ReLU,     Init::He)
        .dense(train.classes(),  Activation::Identity, Init::He)   // logits — softmax is in the loss
        .loss_softmax_cross_entropy()
        .optimizer(std::make_unique<SGD>(cfg.lr))
        .eval_interval(cfg.eval_interval)
        .build();

    // Preallocated once — the training loop allocates nothing.
    Matrix Xb(B, train.features()), Yb(B, train.classes());
    AccuracyMeter meter(B);

    i32 nb = train.num_batches(B);
    RLOG(LL_INFO, "MNIST %dx%d -> %d -> %d | batch %d | lr %.3f | %d epochs | seed %u",
         train.size(), train.features(), cfg.hidden, train.classes(),
         B, cfg.lr, cfg.epochs, cfg.seed);
    RLOG(LL_INFO, "train %d samples (%d batches/epoch), test %d samples",
         train.size(), nb, test.size());

    auto t0 = std::chrono::high_resolution_clock::now();
    f32 best = 0.0f;

    for (i32 e = 1; e <= cfg.epochs; e++) {
        train.shuffle();
        for (i32 i = 0; i < nb; i++) {
            train.batch(i, Xb, Yb);
            net.train_step(Xb, Yb);
        }
        f32 acc = evaluate(net, test, meter, Xb, Yb, B);
        if (acc > best) best = acc;
        RLOG(LL_INFO, "epoch %2d/%d | loss %.4f | test acc %.2f%% (%lld samples)",
             e, cfg.epochs, net.last_loss(), acc * 100.0f, (long long)meter.total());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    f64 secs = std::chrono::duration<f64>(t1 - t0).count();
    f32 final_acc = evaluate(net, test, meter, Xb, Yb, B);
    RLOG(LL_INFO, "done in %.2fs | final test acc %.2f%% | best %.2f%%",
         secs, final_acc * 100.0f, best * 100.0f);
    return 0;
}
