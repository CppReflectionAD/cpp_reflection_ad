// nn/rules.hpp — vector-Jacobian products for the primitives of tensor.hpp.
//
// tensor.hpp stays AD-agnostic; this header is the seam. Each specialization
// says "this function template is a primitive, and here is its VJP": for a
// call y = f(args...) and the adjoint g = dL/dy, pullback<I>(g, y, args...)
// returns dL/d(arg I), with arg I's type (nn::like_as converts where the
// arguments split static and dynamic extents differently). The engine calls it
// only for the arguments that depend on a differentiated input. `reads` says
// which of y and args the pullback uses (like_as and filled_like use an
// argument's shape), so grad() can skip computing the others.
//
// The key is the reflection of the template (`key(^^nn::matvec)`; key() only
// normalises the representation, see autodiff.hpp), not its name: a `matvec`
// in another namespace is not this primitive.

#ifndef NN_RULES_HPP
#define NN_RULES_HPP

#include "autodiff.hpp"
#include "tensor.hpp"

#include <cstddef>

// z = a + b:  ā = z̄, b̄ = z̄
template <> struct nn::ad::vjp<nn::ad::key(^^nn::add)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, const auto &a,
                                 const auto &b) {
    if constexpr (I == 0)
      return nn::like_as(a, g);
    else
      return nn::like_as(b, g);
  }
};

// z = a - b:  ā = z̄, b̄ = -z̄
template <> struct nn::ad::vjp<nn::ad::key(^^nn::sub)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, const auto &a,
                                 const auto &b) {
    if constexpr (I == 0)
      return nn::like_as(a, g);
    else
      return nn::like_as(b, nn::scale(-1.0, g));
  }
};

// z = a ⊙ b:  ā = z̄ ⊙ b, b̄ = z̄ ⊙ a
template <> struct nn::ad::vjp<nn::ad::key(^^nn::hadamard)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, const auto &a,
                                 const auto &b) {
    if constexpr (I == 0)
      return nn::like_as(a, nn::hadamard(g, b));
    else
      return nn::like_as(b, nn::hadamard(g, a));
  }
};

// z = s a:  s̄ = ⟨z̄, a⟩, ā = s z̄
template <> struct nn::ad::vjp<nn::ad::key(^^nn::scale)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, double s,
                                 const auto &a) {
    if constexpr (I == 0)
      return nn::dot(g, a);
    else
      return nn::scale(s, g);
  }
};

// z = relu(a):  ā = z̄ ⊙ [a > 0]
template <> struct nn::ad::vjp<nn::ad::key(^^nn::relu)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, const auto &a) {
    return nn::hadamard(g, nn::relu_mask(a));
  }
};

// y = σ(a):  ā = ȳ ⊙ y(1 - y)   (from the output, not the input)
template <> struct nn::ad::vjp<nn::ad::key(^^nn::sigmoid)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::result;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &y, const auto &) {
    return nn::hadamard(g, nn::sigmoid_slope(y));
  }
};

// s = Σ a:  ā = s̄ 1
template <> struct nn::ad::vjp<nn::ad::key(^^nn::sum)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(double g, double, const auto &a) {
    return nn::filled_like(a, g);
  }
};

// s = ⟨a, b⟩:  ā = s̄ b, b̄ = s̄ a
template <> struct nn::ad::vjp<nn::ad::key(^^nn::dot)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(double g, double, const auto &a,
                                 const auto &b) {
    if constexpr (I == 0)
      return nn::like_as(a, nn::scale(g, b));
    else
      return nn::like_as(b, nn::scale(g, a));
  }
};

// s = Σ (y - t)² / n:  ȳ = s̄ 2(y - t)/n, t̄ = -ȳ
template <> struct nn::ad::vjp<nn::ad::key(^^nn::mse)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(double g, double, const auto &y,
                                 const auto &t) {
    const double k = 2.0 * g / static_cast<double>(y.d.size());
    if constexpr (I == 0)
      return nn::like_as(y, nn::scale(k, nn::sub(y, t)));
    else
      return nn::like_as(t, nn::scale(-k, nn::sub(y, t)));
  }
};

// z = W x:  W̄ = z̄ xᵀ, x̄ = Wᵀ z̄
template <> struct nn::ad::vjp<nn::ad::key(^^nn::matvec)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, const auto &W,
                                 const auto &x) {
    if constexpr (I == 0)
      return nn::like_as(W, nn::outer(g, x));
    else
      return nn::like_as(x, nn::matvec_transposed(W, g));
  }
};

// Z = A B:  Ā = Z̄ Bᵀ, B̄ = Aᵀ Z̄
template <> struct nn::ad::vjp<nn::ad::key(^^nn::matmul)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, const auto &A,
                                 const auto &B) {
    if constexpr (I == 0)
      return nn::like_as(A, nn::matmul(g, nn::transpose(B)));
    else
      return nn::like_as(B, nn::matmul(nn::transpose(A), g));
  }
};

// Z = Aᵀ:  Ā = Z̄ᵀ
template <> struct nn::ad::vjp<nn::ad::key(^^nn::transpose)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::nothing;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, const auto &) {
    return nn::transpose(g);
  }
};

// Z = u vᵀ:  ū = Z̄ v, v̄ = Z̄ᵀ u
template <> struct nn::ad::vjp<nn::ad::key(^^nn::outer)> {
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &, const auto &u,
                                 const auto &v) {
    if constexpr (I == 0)
      return nn::like_as(u, nn::matvec(g, v));
    else
      return nn::like_as(v, nn::matvec_transposed(g, u));
  }
};

#endif // NN_RULES_HPP
