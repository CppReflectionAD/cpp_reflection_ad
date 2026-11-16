// ad_tensor_operator_by_value.cpp — overloaded operators taking a tensor by
// value in a reflected tensor body.
//
// Each operand of `x * W` is then a copy of the argument: a conversion that
// constructs the parameter, which the engine has to see through to reach the
// name (#248). Otherwise as in ad_tensor_operator.cpp, which takes `const
// Tensor &`.

#include <test_simple_include.hpp>

#include <ad/autograd_tensor.h>
#include <ad/tensor.h>
#include <ad/tensor_rules.h>

#include <cmath>
#include <cstddef>

namespace mat {
using toy_tensor::Tensor;

inline Tensor operator*(Tensor a, Tensor b) { return toy_tensor::matmul(a, b); }
inline Tensor operator+(Tensor a, Tensor b) { return toy_tensor::add(a, b); }

inline Tensor loss(Tensor x, Tensor W, Tensor b) {
  Tensor z = x * W;
  Tensor y = z + b;
  Tensor sq = toy_tensor::mul(y, y);
  Tensor L = toy_tensor::sum(sq);
  return L;
}
} // namespace mat

template <> struct ad::primitive<^^mat::operator*> {
  static constexpr ad::OpKind op = ad::OpKind::Matmul;
};

int main() {
  using toy_tensor::Tensor;
  Tensor x(2, 2), W(2, 2), b(1, 2);
  auto fill = [](Tensor &t, double s) {
    for (std::size_t i = 0; i < t.d.size(); ++i)
      t.d[i] = std::sin(s + 0.37 * i) * 0.5;
  };
  fill(x, 0.1);
  fill(W, 1.0);
  fill(b, 2.0);

  auto grads = ad::gradient_wrt<^^mat::loss, Tensor, 1, 2>(x, W, b);

  // Against central differences of the loss itself.
  Tensor *P[2] = {&W, &b};
  const double h = 1e-6;
  for (int k = 0; k < 2; ++k)
    for (std::size_t i = 0; i < P[k]->d.size(); ++i) {
      const double orig = P[k]->d[i];
      P[k]->d[i] = orig + h;
      const double Lp = mat::loss(x, W, b).d[0];
      P[k]->d[i] = orig - h;
      const double Lm = mat::loss(x, W, b).d[0];
      P[k]->d[i] = orig;
      EXPECT_NEAR_ABS(grads[k].d[i], (Lp - Lm) / (2 * h), 1e-6);
    }

  TEST_END;
}
