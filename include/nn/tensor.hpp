// nn/tensor.hpp — fixed-size vector / matrix values and the primitive ops a
// network is written with.
//
// Nothing here knows about differentiation: rules.hpp registers a VJP for each
// primitive, and autodiff.hpp differentiates code that calls them.
//
// Design constraints, all coming from what body reflection can see today:
//
//   - Shapes are template parameters and the types are trivially copyable
//     (std::array storage, no heap), so shape errors are compile errors. The
//     engine also handles heap-backed types (tests/common/nn_mlp.t.cpp has
//     one).
//   - Values, not in-place updates. Reflected code is `T v = expr;` and
//     `return expr;`; assignments and compound assignments are not modelled.
//     (sgd_update below mutates, but it runs outside reflected code.)
//   - Each primitive is a single function template with a unique name, so its
//     rule can be keyed on `^^nn::matvec` etc. The operators (`W * x`,
//     `a + b`, ...) are one-line wrappers over the primitives: they have no
//     rule, so the engine inlines their bodies and finds the primitive inside.
//   - No expression templates: an operator returns a value, never a proxy.

#ifndef NN_TENSOR_HPP
#define NN_TENSOR_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace nn {

template <std::size_t N> struct Vec {
  static constexpr std::size_t size = N;
  std::array<double, N> d{};

  constexpr double &operator[](std::size_t i) { return d[i]; }
  constexpr const double &operator[](std::size_t i) const { return d[i]; }
};

// Row-major.
template <std::size_t R, std::size_t C> struct Mat {
  static constexpr std::size_t rows = R;
  static constexpr std::size_t cols = C;
  static constexpr std::size_t size = R * C;
  std::array<double, R * C> d{};

  constexpr double &operator()(std::size_t i, std::size_t j) {
    return d[i * C + j];
  }
  constexpr const double &operator()(std::size_t i, std::size_t j) const {
    return d[i * C + j];
  }
};

// Vec and Mat: a flat array `d` of `size` doubles. Elementwise ops work on any
// such type.
template <class T>
concept tensor = requires(T t) {
  { T::size } -> std::convertible_to<std::size_t>;
  { t.d[0] } -> std::convertible_to<double>;
};

// ---------------------------------------------------------------------------
// Primitives. Each has a VJP rule in rules.hpp; keep the names unique (one
// template per name) so `^^nn::<name>` reflects exactly that template.
// ---------------------------------------------------------------------------

template <tensor T> constexpr T add(const T &a, const T &b) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = a.d[i] + b.d[i];
  return o;
}

template <tensor T> constexpr T sub(const T &a, const T &b) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = a.d[i] - b.d[i];
  return o;
}

// Elementwise product.
template <tensor T> constexpr T hadamard(const T &a, const T &b) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = a.d[i] * b.d[i];
  return o;
}

template <tensor T> constexpr T scale(double s, const T &a) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = s * a.d[i];
  return o;
}

template <tensor T> constexpr T relu(const T &a) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = a.d[i] > 0.0 ? a.d[i] : 0.0;
  return o;
}

template <tensor T> T sigmoid(const T &a) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = 1.0 / (1.0 + std::exp(-a.d[i]));
  return o;
}

template <tensor T> constexpr double sum(const T &a) {
  double s = 0.0;
  for (std::size_t i = 0; i < T::size; ++i)
    s += a.d[i];
  return s;
}

template <tensor T> constexpr double dot(const T &a, const T &b) {
  double s = 0.0;
  for (std::size_t i = 0; i < T::size; ++i)
    s += a.d[i] * b.d[i];
  return s;
}

// Mean squared error.
template <tensor T> constexpr double mse(const T &y, const T &t) {
  double s = 0.0;
  for (std::size_t i = 0; i < T::size; ++i)
    s += (y.d[i] - t.d[i]) * (y.d[i] - t.d[i]);
  return s / static_cast<double>(T::size);
}

// W x
template <std::size_t R, std::size_t C>
constexpr Vec<R> matvec(const Mat<R, C> &W, const Vec<C> &x) {
  Vec<R> o;
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      o.d[i] += W(i, j) * x.d[j];
  return o;
}

// A B
template <std::size_t R, std::size_t K, std::size_t C>
constexpr Mat<R, C> matmul(const Mat<R, K> &A, const Mat<K, C> &B) {
  Mat<R, C> o;
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t k = 0; k < K; ++k)
      for (std::size_t j = 0; j < C; ++j)
        o(i, j) += A(i, k) * B(k, j);
  return o;
}

template <std::size_t R, std::size_t C>
constexpr Mat<C, R> transpose(const Mat<R, C> &A) {
  Mat<C, R> o;
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      o(j, i) = A(i, j);
  return o;
}

// u vᵀ
template <std::size_t R, std::size_t C>
constexpr Mat<R, C> outer(const Vec<R> &u, const Vec<C> &v) {
  Mat<R, C> o;
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      o(i, j) = u.d[i] * v.d[j];
  return o;
}

// ---------------------------------------------------------------------------
// Operators: plain wrappers, no rules. The engine inlines them.
// ---------------------------------------------------------------------------

template <std::size_t R, std::size_t C>
constexpr Vec<R> operator*(const Mat<R, C> &W, const Vec<C> &x) {
  return matvec(W, x);
}

template <std::size_t R, std::size_t K, std::size_t C>
constexpr Mat<R, C> operator*(const Mat<R, K> &A, const Mat<K, C> &B) {
  return matmul(A, B);
}

template <tensor T> constexpr T operator*(double s, const T &a) {
  return scale(s, a);
}

template <tensor T> constexpr T operator+(const T &a, const T &b) {
  return add(a, b);
}

template <tensor T> constexpr T operator-(const T &a, const T &b) {
  return sub(a, b);
}

// ---------------------------------------------------------------------------
// Helpers for the rules and for training loops. Not differentiable: they are
// never called from reflected code.
// ---------------------------------------------------------------------------

// Wᵀ g, without forming Wᵀ.
template <std::size_t R, std::size_t C>
constexpr Vec<C> matvec_transposed(const Mat<R, C> &W, const Vec<R> &g) {
  Vec<C> o;
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      o.d[j] += W(i, j) * g.d[i];
  return o;
}

// 1 where a > 0, else 0: relu's derivative (0 at the kink).
template <tensor T> constexpr T relu_mask(const T &a) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = a.d[i] > 0.0 ? 1.0 : 0.0;
  return o;
}

// σ'(a) from y = σ(a): y (1 - y).
template <tensor T> constexpr T sigmoid_slope(const T &y) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = y.d[i] * (1.0 - y.d[i]);
  return o;
}

template <tensor T> constexpr T filled(double v) {
  T o;
  for (std::size_t i = 0; i < T::size; ++i)
    o.d[i] = v;
  return o;
}

// p -= lr * g
template <tensor T> constexpr void sgd_update(T &p, const T &g, double lr) {
  for (std::size_t i = 0; i < T::size; ++i)
    p.d[i] -= lr * g.d[i];
}

} // namespace nn

#endif // NN_TENSOR_HPP
