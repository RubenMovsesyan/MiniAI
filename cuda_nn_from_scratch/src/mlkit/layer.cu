#include <rlog.h>
#include <mlkit/layer.cuh>
#include <mlkit/init.cuh>
#include <nn/activations.cuh>
#include <nn/activation_gradients.cuh>
#include <agg/agg.cuh>

Dense::Dense(i32 batch, i32 in, i32 out, Activation activation, Init init)
    : W(in, out), b(1, out), dW(in, out), db(1, out),
      Z(batch, out), A(batch, out), dZ(batch, out), dX(batch, in),
      Xt(in, batch), Wt(out, in), dW_grad(in, out), db_grad(1, out),
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

// Y = X·W + b, then activation.
// Every op writes into a preallocated buffer, so no cudaMalloc/cudaFree runs here and
// nothing synchronizes: the kernels queue on g_compute_stream and the CPU races ahead.
// (An expression like `Z = X.ref().mul(W.ref()).rowAdd(...)` would put the matmul as an
// inner node, forcing materialize() to malloc+free a temp — a device-wide sync per call.)
const Matrix& Dense::forward(const Matrix& X) {
    input_cache = &X;
    X.matmul(W, Z);                     // GEMM straight into Z — operands are plain refs, no temp
    Z = Z.ref().rowAdd(b.ref());        // element-wise into Z (each thread reads/writes its own cell)
    if (act == Activation::ReLU) {
        relu(Z, A);                     // out-param
        return A;
    }
    return Z;                           // identity: logits are the pre-activation
}

const Matrix& Dense::backward(const Matrix& dA) {
    // dZ = act'(Z) ⊙ dA  (identity → dZ is dA itself)
    const Matrix* dZp;
    if (act == Activation::ReLU) {
        grad_relu(Z, dA, dZ);           // out-param; grad_relu masks by the pre-activation Z
        dZp = &dZ;
    } else {
        dZp = &dA;
    }
    const Matrix& g = *dZp;

    // dW += Xᵀ·dZ ; db += col_sum(dZ)  (accumulate) — all through preallocated scratch
    input_cache->transposed(Xt);        // Xt = Xᵀ
    Xt.matmul(g, dW_grad);              // dW_grad = Xᵀ·dZ
    dW = dW.ref() + dW_grad.ref();
    col_sum(g, db_grad);                // out-param
    db = db.ref() + db_grad.ref();

    // dX = dZ·Wᵀ  (handed to the previous layer)
    W.transposed(Wt);
    g.matmul(Wt, dX);
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
