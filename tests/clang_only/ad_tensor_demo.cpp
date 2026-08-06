// ad_tensor_demo.cpp — reflection AD on TENSOR / ML code.
//
// The same reflection engine handles the ML case: a small neural net whose
// parameters are tensors, differentiated w.r.t. the *parameters* (not the data).
// The tensor type + ops live in tensor.h; the engine names no concrete type.
//
// Build/run: `make run-tensor`.

#include "autograd_tensor.h"
#include "tensor.h"
#include <cmath>
#include <cstdio>

static double loss_scalar(const nn::Tensor &x, const nn::Tensor &W1, const nn::Tensor &b1,
                          const nn::Tensor &W2, const nn::Tensor &b2) {
  return nn::loss(x, W1, b1, W2, b2).d[0];
}

int main() {
  using nn::Tensor;
  int B = 4, In = 3, H = 5, Out = 2;
  Tensor x(B, In), W1(In, H), b1(1, H), W2(H, Out), b2(1, Out);
  auto fill = [](Tensor &t, double s) {
    for (std::size_t i = 0; i < t.d.size(); ++i) t.d[i] = std::sin(s + 0.37 * i) * 0.5;
  };
  fill(x, 0.1); fill(W1, 1.0); fill(b1, 2.0); fill(W2, 3.0); fill(b2, 4.0);

  // Gradients w.r.t. parameters (indices 1..4), NOT data x (0).
  auto grads = ad::gradient_wrt<^^nn::loss, Tensor, 1, 2, 3, 4>(x, W1, b1, W2, b2);

  std::printf("=== reflection AD on a 1-hidden-layer MLP ===\n");
  std::printf("loss = %.6f   (batch=%d, in=%d, hidden=%d, out=%d)\n",
              loss_scalar(x, W1, b1, W2, b2), B, In, H, Out);
  std::printf("parameter gradients via gradient_wrt<^^loss, W1,b1,W2,b2>; "
              "data x is pruned by activity analysis.\n\n");

  Tensor *P[4]   = {&W1, &b1, &W2, &b2};
  const char *nm[4] = {"W1", "b1", "W2", "b2"};
  double h = 1e-6, maxrel = 0.0; int fails = 0, total = 0;
  for (int k = 0; k < 4; ++k) {
    for (std::size_t idx = 0; idx < P[k]->d.size(); ++idx) {
      double orig = P[k]->d[idx];
      P[k]->d[idx] = orig + h; double Lp = loss_scalar(x, W1, b1, W2, b2);
      P[k]->d[idx] = orig - h; double Lm = loss_scalar(x, W1, b1, W2, b2);
      P[k]->d[idx] = orig;
      double rel = std::abs((Lp - Lm) / (2 * h) - grads[k].d[idx]) / (1.0 + std::abs(grads[k].d[idx]));
      maxrel = std::max(maxrel, rel); ++total; fails += (rel > 1e-6);
    }
    std::printf("  d/d%-2s : %2d values, e.g. grad[0]=% .6f\n",
                nm[k], (int)P[k]->d.size(), grads[k].d[0]);
  }
  std::printf("\nchecked %d parameter-gradient values vs finite differences\n", total);
  std::printf("max relative error: %.2e  ->  %s\n", maxrel, fails ? "MISMATCH" : "ALL MATCH");
  return fails ? 1 : 0;
}
