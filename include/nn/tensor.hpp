// nn/tensor.hpp — vector and matrix values, each dimension static or dynamic,
// and the primitive ops a network is written with.
//
// Nothing here knows about differentiation: rules.hpp gives a rule (a VJP) for
// each primitive, and autodiff.hpp differentiates code that calls them.
//
// Design constraints, all coming from what body reflection can see today:
//
//   - A tensor's shape is a std::extents. With every extent static (Vec<3>,
//     Mat<2, 3>) the storage is a std::array: trivially copyable, usable as a
//     constexpr constant. Any std::dynamic_extent (`nn::dyn`) makes it a
//     std::vector. Ops combine static and dynamic extents (Vec<3> + Vec<dyn>
//     is a Vec<3>). Two static extents that disagree are a compile error; a
//     mismatch involving a dynamic one throws std::invalid_argument.
//   - Values, not in-place updates. Reflected code is `T v = expr;` and
//     `return expr;`; assignments and compound assignments are not modelled.
//     (sgd_update below mutates, but it runs outside reflected code.)
//   - Each primitive is a single function template with a unique name, so its
//     rule can list `^^nn::matvec` etc. The operators (`W * x`,
//     `a + b`, ...) are one-line wrappers over the primitives: they have no
//     rule, so the engine inlines their bodies and finds the primitive inside.
//   - No expression templates: an operator returns a value, never a proxy.

#ifndef NN_TENSOR_HPP
#define NN_TENSOR_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <mdspan>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nn {

inline constexpr std::size_t dyn = std::dynamic_extent;

namespace detail {

struct no_extents {};

// The element count of a fully static shape, else 0.
template <class E> consteval std::size_t static_size() {
  if constexpr (E::rank_dynamic() > 0)
    return 0;
  std::size_t n = 1;
  for (std::size_t r = 0; r < E::rank(); ++r)
    n *= E::static_extent(r);
  return n;
}

template <class E>
using storage_t = std::conditional_t<E::rank_dynamic() == 0,
                                     std::array<double, static_size<E>()>,
                                     std::vector<double>>;

template <class E>
using dims_t = std::conditional_t<E::rank_dynamic() == 0, no_extents,
                                  std::array<std::size_t, E::rank_dynamic()>>;

constexpr void check(bool same_shape) {
  if (!same_shape)
    throw std::invalid_argument("nn: shape mismatch");
}

} // namespace detail

// Row-major. `dims` holds only the dynamic extents, so a fully static tensor
// is an aggregate of its elements: `Vec<2>{{1.0, 2.0}}`.
template <class E> struct Tensor {
  using extents_type = E;
  detail::storage_t<E> d{};
  detail::dims_t<E> dims{};

  constexpr E extents() const {
    if constexpr (E::rank_dynamic() == 0)
      return E{};
    else
      return E(dims);
  }
  constexpr auto view() { return std::mdspan(d.data(), extents()); }
  constexpr auto view() const { return std::mdspan(d.data(), extents()); }
  constexpr double &operator[](auto... i) { return view()[i...]; }
  constexpr const double &operator[](auto... i) const { return view()[i...]; }
};

template <std::size_t N> using Vec = Tensor<std::extents<std::size_t, N>>;
template <std::size_t R, std::size_t C>
using Mat = Tensor<std::extents<std::size_t, R, C>>;

// A flat array `d` of doubles with a std::extents shape. Elementwise ops work
// on any such type.
template <class T>
concept tensor = requires(const T &t) {
  typename T::extents_type;
  { t.extents() } -> std::same_as<typename T::extents_type>;
  { t.d.size() } -> std::convertible_to<std::size_t>;
  { t.d[0] } -> std::convertible_to<double>;
};
template <class T>
concept vector_like = tensor<T> && T::extents_type::rank() == 1;
template <class T>
concept matrix_like = tensor<T> && T::extents_type::rank() == 2;

// Extent R of T's type: a size, or dyn.
template <tensor T, std::size_t R>
inline constexpr std::size_t extent_v = T::extents_type::static_extent(R);

// A zero tensor of type T, given its dynamic extents (or all its extents).
template <tensor T, class... I> constexpr T zeros(I... sizes) {
  using E = typename T::extents_type;
  static_assert(sizeof...(I) == E::rank_dynamic() || sizeof...(I) == E::rank(),
                "nn::zeros: give T's dynamic extents, or all of them");
  if constexpr (sizeof...(I) == E::rank() && E::rank() > 0) {
    const std::array<std::size_t, E::rank()> all{
        static_cast<std::size_t>(sizes)...};
    for (std::size_t r = 0; r < E::rank(); ++r)
      detail::check(E::static_extent(r) == dyn ||
                    E::static_extent(r) == all[r]);
  }
  const E ext(static_cast<std::size_t>(sizes)...);
  T t{};
  if constexpr (E::rank_dynamic() > 0) {
    std::size_t n = 1;
    for (std::size_t r = 0, k = 0; r < E::rank(); ++r) {
      n *= ext.extent(r);
      if (E::static_extent(r) == dyn)
        t.dims[k++] = ext.extent(r);
    }
    t.d.assign(n, 0.0);
  }
  return t;
}

// A zero tensor of a's shape. The engine also finds it by ADL, for the
// gradient of a parameter that no adjoint reaches.
template <tensor T> constexpr T zeros_like(const T &a) {
  T t{};
  t.dims = a.dims;
  if constexpr (T::extents_type::rank_dynamic() > 0)
    t.d.assign(a.d.size(), 0.0);
  return t;
}

namespace detail {

template <std::size_t A, std::size_t B>
inline constexpr bool compatible = A == dyn || B == dyn || A == B;

// Same rank, and no two static extents that disagree.
template <class A, class B> consteval bool compatible_shapes() {
  using EA = typename A::extents_type;
  using EB = typename B::extents_type;
  if (EA::rank() != EB::rank())
    return false;
  for (std::size_t r = 0; r < EA::rank(); ++r)
    if (EA::static_extent(r) != dyn && EB::static_extent(r) != dyn &&
        EA::static_extent(r) != EB::static_extent(r))
      return false;
  return true;
}

template <class EA, class EB, std::size_t... R>
auto common_extents(std::index_sequence<R...>)
    -> std::extents<std::size_t,
                    (EA::static_extent(R) != dyn ? EA::static_extent(R)
                                                 : EB::static_extent(R))...>;

template <class A, class B> constexpr void same_shape(const A &a, const B &b) {
  check(a.extents() == b.extents());
}

} // namespace detail

// Tensors an elementwise op can combine: Vec<3> with Vec<dyn>, not with Vec<4>.
template <class A, class B>
concept same_shape_as =
    tensor<A> && tensor<B> && detail::compatible_shapes<A, B>();

// The result of an elementwise op on A and B: each extent static if either is.
template <class A, class B>
using common_t =
    Tensor<decltype(detail::common_extents<typename A::extents_type,
                                           typename B::extents_type>(
        std::make_index_sequence<A::extents_type::rank()>{}))>;

// A zero T with the extents `ext`, which may split static and dynamic
// differently.
template <tensor T, class E> constexpr T zeros_shaped(const E &ext) {
  return [&]<std::size_t... R>(std::index_sequence<R...>) {
    return zeros<T>(ext.extent(R)...);
  }(std::make_index_sequence<E::rank()>{});
}

// g's values as a tensor of a's type (g has a's shape): a rule returns the
// adjoint of an argument in that argument's type.
template <tensor A, same_shape_as<A> G>
constexpr A like_as(const A &a, const G &g) {
  detail::same_shape(a, g);
  if constexpr (std::is_same_v<A, G>) {
    return g;
  } else {
    A o = zeros_shaped<A>(g.extents());
    for (std::size_t i = 0; i < g.d.size(); ++i)
      o.d[i] = g.d[i];
    return o;
  }
}

// ---------------------------------------------------------------------------
// Primitives. Each has a VJP rule in rules.hpp; keep the names unique (one
// template per name) so `^^nn::<name>` reflects exactly that template.
// ---------------------------------------------------------------------------

template <tensor A, same_shape_as<A> B>
constexpr common_t<A, B> add(const A &a, const B &b) {
  detail::same_shape(a, b);
  auto o = zeros_shaped<common_t<A, B>>(a.extents());
  for (std::size_t i = 0; i < a.d.size(); ++i)
    o.d[i] = a.d[i] + b.d[i];
  return o;
}

template <tensor A, same_shape_as<A> B>
constexpr common_t<A, B> sub(const A &a, const B &b) {
  detail::same_shape(a, b);
  auto o = zeros_shaped<common_t<A, B>>(a.extents());
  for (std::size_t i = 0; i < a.d.size(); ++i)
    o.d[i] = a.d[i] - b.d[i];
  return o;
}

// Elementwise product.
template <tensor A, same_shape_as<A> B>
constexpr common_t<A, B> hadamard(const A &a, const B &b) {
  detail::same_shape(a, b);
  auto o = zeros_shaped<common_t<A, B>>(a.extents());
  for (std::size_t i = 0; i < a.d.size(); ++i)
    o.d[i] = a.d[i] * b.d[i];
  return o;
}

template <tensor T> constexpr T scale(double s, const T &a) {
  T o = zeros_like(a);
  for (std::size_t i = 0; i < a.d.size(); ++i)
    o.d[i] = s * a.d[i];
  return o;
}

template <tensor T> constexpr T relu(const T &a) {
  T o = zeros_like(a);
  for (std::size_t i = 0; i < a.d.size(); ++i)
    o.d[i] = a.d[i] > 0.0 ? a.d[i] : 0.0;
  return o;
}

template <tensor T> T sigmoid(const T &a) {
  T o = zeros_like(a);
  for (std::size_t i = 0; i < a.d.size(); ++i)
    o.d[i] = 1.0 / (1.0 + std::exp(-a.d[i]));
  return o;
}

template <tensor T> constexpr double sum(const T &a) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.d.size(); ++i)
    s += a.d[i];
  return s;
}

template <tensor A, same_shape_as<A> B>
constexpr double dot(const A &a, const B &b) {
  detail::same_shape(a, b);
  double s = 0.0;
  for (std::size_t i = 0; i < a.d.size(); ++i)
    s += a.d[i] * b.d[i];
  return s;
}

// Mean squared error.
template <tensor Y, same_shape_as<Y> T>
constexpr double mse(const Y &y, const T &t) {
  detail::same_shape(y, t);
  double s = 0.0;
  for (std::size_t i = 0; i < y.d.size(); ++i)
    s += (y.d[i] - t.d[i]) * (y.d[i] - t.d[i]);
  return s / static_cast<double>(y.d.size());
}

// W x
template <matrix_like W, vector_like X>
constexpr Vec<extent_v<W, 0>> matvec(const W &w, const X &x) {
  static_assert(detail::compatible<extent_v<W, 1>, extent_v<X, 0>>,
                "nn::matvec: W has as many columns as x has elements");
  const std::size_t rows = w.extents().extent(0), cols = w.extents().extent(1);
  detail::check(cols == x.extents().extent(0));
  auto o = zeros<Vec<extent_v<W, 0>>>(rows);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      o.d[i] += w[i, j] * x.d[j];
  return o;
}

// A B
template <matrix_like A, matrix_like B>
constexpr Mat<extent_v<A, 0>, extent_v<B, 1>> matmul(const A &a, const B &b) {
  static_assert(detail::compatible<extent_v<A, 1>, extent_v<B, 0>>,
                "nn::matmul: A has as many columns as B has rows");
  const std::size_t rows = a.extents().extent(0), inner = a.extents().extent(1),
                    cols = b.extents().extent(1);
  detail::check(inner == b.extents().extent(0));
  auto o = zeros<Mat<extent_v<A, 0>, extent_v<B, 1>>>(rows, cols);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t k = 0; k < inner; ++k)
      for (std::size_t j = 0; j < cols; ++j)
        o[i, j] += a[i, k] * b[k, j];
  return o;
}

template <matrix_like A>
constexpr Mat<extent_v<A, 1>, extent_v<A, 0>> transpose(const A &a) {
  const std::size_t rows = a.extents().extent(0), cols = a.extents().extent(1);
  auto o = zeros<Mat<extent_v<A, 1>, extent_v<A, 0>>>(cols, rows);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      o[j, i] = a[i, j];
  return o;
}

// u vᵀ
template <vector_like U, vector_like V>
constexpr Mat<extent_v<U, 0>, extent_v<V, 0>> outer(const U &u, const V &v) {
  auto o = zeros<Mat<extent_v<U, 0>, extent_v<V, 0>>>(u.d.size(), v.d.size());
  for (std::size_t i = 0; i < u.d.size(); ++i)
    for (std::size_t j = 0; j < v.d.size(); ++j)
      o[i, j] = u.d[i] * v.d[j];
  return o;
}

// ---------------------------------------------------------------------------
// Operators: plain wrappers, no rules. The engine inlines them.
// ---------------------------------------------------------------------------

template <matrix_like W, vector_like X>
constexpr Vec<extent_v<W, 0>> operator*(const W &w, const X &x) {
  return matvec(w, x);
}

template <matrix_like A, matrix_like B>
constexpr Mat<extent_v<A, 0>, extent_v<B, 1>> operator*(const A &a,
                                                        const B &b) {
  return matmul(a, b);
}

template <tensor T> constexpr T operator*(double s, const T &a) {
  return scale(s, a);
}

template <tensor A, same_shape_as<A> B>
constexpr common_t<A, B> operator+(const A &a, const B &b) {
  return add(a, b);
}

template <tensor A, same_shape_as<A> B>
constexpr common_t<A, B> operator-(const A &a, const B &b) {
  return sub(a, b);
}

// ---------------------------------------------------------------------------
// Helpers for the rules and for training loops. Not differentiable: they are
// never called from reflected code.
// ---------------------------------------------------------------------------

// Wᵀ g, without forming Wᵀ.
template <matrix_like W, vector_like G>
constexpr Vec<extent_v<W, 1>> matvec_transposed(const W &w, const G &g) {
  const std::size_t rows = w.extents().extent(0), cols = w.extents().extent(1);
  detail::check(rows == g.extents().extent(0));
  auto o = zeros<Vec<extent_v<W, 1>>>(cols);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      o.d[j] += w[i, j] * g.d[i];
  return o;
}

// 1 where a > 0, else 0: relu's derivative (0 at the kink).
template <tensor T> constexpr T relu_mask(const T &a) {
  T o = zeros_like(a);
  for (std::size_t i = 0; i < a.d.size(); ++i)
    o.d[i] = a.d[i] > 0.0 ? 1.0 : 0.0;
  return o;
}

// σ'(a) from y = σ(a): y (1 - y).
template <tensor T> constexpr T sigmoid_slope(const T &y) {
  T o = zeros_like(y);
  for (std::size_t i = 0; i < y.d.size(); ++i)
    o.d[i] = y.d[i] * (1.0 - y.d[i]);
  return o;
}

// A tensor of a's shape with every element v.
template <tensor T> constexpr T filled_like(const T &a, double v) {
  T o = zeros_like(a);
  for (double &x : o.d)
    x = v;
  return o;
}

// A T with every element v, given its dynamic extents (or all of them).
template <tensor T, class... I> constexpr T filled(double v, I... sizes) {
  return filled_like(zeros<T>(sizes...), v);
}

// p -= lr * g
template <tensor T> constexpr void sgd_update(T &p, const T &g, double lr) {
  detail::same_shape(p, g);
  for (std::size_t i = 0; i < p.d.size(); ++i)
    p.d[i] -= lr * g.d[i];
}

} // namespace nn

#endif // NN_TENSOR_HPP
