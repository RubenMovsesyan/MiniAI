#include <rlog.h>
#include <mlkit/layer.cuh>
#include <mlkit/init.cuh>
#include <nn/activations.cuh>
#include <nn/activation_gradients.cuh>
#include <agg/agg.cuh>

Dense::Dense(i32 batch, i32 in, i32 out, Activation activation, Init init)
    : W(in, out), b(1, out), dW(in, out), db(1, out),
      Z(batch, out), A(batch, out), dZ(batch, out), dX(batch, in),
      act(activation) {
    switch (init) {
        case Init::He:     he_normal(W, in);          break;
        case Init::LeCun:  lecun_normal(W, in);        break;
        case Init::Xavier: xavier_normal(W, in, out);  break;
    }
    zero_init(b);
    zero_init(dW);
    zero_init(db);
}

// Y = X·W + b, then activation. Matmul materializes, rowAdd fuses into one kernel.
const Matrix& Dense::forward(const Matrix& X) {
    input_cache = &X;
    Z = X.ref().mul(W.ref()).rowAdd(b.ref());
    if (act == Activation::ReLU) {
        relu(Z, A);          // out-param: no allocation
        return A;
    }
    return Z;                // identity: logits are the pre-activation
}

const Matrix& Dense::backward(const Matrix& dA) {
    // dZ = act'(Z) ⊙ dA  (identity → dZ is dA itself)
    const Matrix* dZp;
    if (act == Activation::ReLU) {
        grad_relu(Z, dA, dZ);   // out-param; grad_relu masks by the pre-activation Z
        dZp = &dZ;
    } else {
        dZp = &dA;
    }
    const Matrix& g = *dZp;

    // dW += Xᵀ·dZ ; db += col_sum(dZ)  (accumulate)
    dW = dW.ref() + input_cache->ref().transpose().mul(g.ref());
    db = db.ref() + col_sum(g);

    // dX = dZ·Wᵀ  (handed to the previous layer)
    dX = g.ref().mul(W.ref().transpose());
    return dX;
}

void Dense::zero_grad() {
    zero_init(dW);
    zero_init(db);
}

void Dense::update(Optimizer& opt) {
    opt.update(W, dW);
    opt.update(b, db);
}
