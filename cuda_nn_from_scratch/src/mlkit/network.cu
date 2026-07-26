#include <rlog.h>
#include <mlkit/network.cuh>

Network::Network(std::vector<std::unique_ptr<Layer>> layers, i32 batch, i32 classes,
                 std::unique_ptr<Optimizer> opt, i32 eval_interval)
    : layers_(std::move(layers)), loss_(batch, classes),
      opt_(std::move(opt)), eval_interval_(eval_interval) {}

const Matrix& Network::forward(const Matrix& X) {
    const Matrix* a = &X;
    for (auto& l : layers_) a = &l->forward(*a);
    return *a;
}

void Network::train_step(const Matrix& X, const Matrix& Y) {
    const Matrix& logits = forward(X);

    // Gradient w.r.t. logits from the loss (softmax lives inside it).
    const Matrix* dA = &loss_.backward(logits, Y);

    // Fresh accumulation each step, then walk layers in reverse.
    for (auto& l : layers_) l->zero_grad();
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        dA = &(*it)->backward(*dA);

    for (auto& l : layers_) l->update(*opt_);

    if (eval_interval_ > 0 && (++step_count_ % (u64)eval_interval_) == 0)
        last_loss_ = loss_.value(logits, Y);   // logits buffer untouched by backward
}

NetworkBuilder& NetworkBuilder::dense(i32 units, Activation act, Init init) {
    layers_.push_back(std::make_unique<Dense>(batch_, cur_dim_, units, act, init));
    cur_dim_ = units;
    classes_ = units;   // last dense width = number of output classes
    return *this;
}

Network NetworkBuilder::build() {
    if (!output_layer_ok())
        RLOG(LL_ERROR, "final layer must use Identity activation — softmax lives in the loss "
                       "(guarding against double-softmax)");
    if (!opt_)
        RLOG(LL_ERROR, "no optimizer set on NetworkBuilder");
    return Network(std::move(layers_), batch_, classes_, std::move(opt_), eval_interval_);
}
