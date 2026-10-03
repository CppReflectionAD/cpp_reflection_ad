// cx_sqrt.hpp — constexpr sqrt via range reduction and Newton-Raphson.
//
// Algorithm
// ---------
// 1. Range reduction: x = m · 4^e with m ∈ [1, 4), so sqrt(x) = sqrt(m) · 2^e.
//    Scaling by powers of two is exact.
// 2. Newton-Raphson on m: given estimate yₙ, the next estimate is
//      yₙ₊₁ = 0.5 * (yₙ + m / yₙ)
//    Starting from y₀ = (m + 1) / 2 ≥ sqrt(m), the estimates decrease until
//    they reach machine precision; iteration stops when one no longer does.
//    m ∈ [1, 4) bounds this to a handful of steps.
// 3. Rounding: Newton's own rounding can stop an ulp from the nearest double.
//    y's neighbour y' is nearer sqrt(m) exactly when m is past y·y' (on the
//    far side from y): the midpoint squared is y·y' + ulp²/4, and that last
//    term is below the spacing of m and y·y'. The products are compared
//    exactly, so the result is correctly rounded, as IEEE requires of
//    std::sqrt.
//
// Special cases: negative → NaN, +∞ → +∞, 0 → 0.
//
// Accuracy: identical to std::sqrt (see tests/cx_std/cx_std.t.cpp).

#ifndef CX_STD_CX_SQRT_HPP
#define CX_STD_CX_SQRT_HPP

#include "cx_exp.hpp"

#include <limits>
#include <type_traits>

namespace cx {
namespace detail {

// The neighbours of y ∈ [1, 2]. Below 2 the spacing is epsilon; above it,
// and below 1, it changes by a factor of 2.
template <typename T> constexpr T sqrt_pred(T y) {
  constexpr T eps = std::numeric_limits<T>::epsilon();
  return y == T(1) ? T(1) - eps / T(2) : y - eps;
}
template <typename T> constexpr T sqrt_succ(T y) {
  constexpr T eps = std::numeric_limits<T>::epsilon();
  return y == T(2) ? T(2) + T(2) * eps : y + eps;
}

// m ≤ a·b, exactly, for a·b within a factor of 2 of m. a·b = p + q exactly
// (Dekker's product, splitting each factor in half by Veltkamp), and m - p is
// exact by Sterbenz's lemma.
template <typename T> constexpr bool exact_le(T m, T a, T b) {
  const auto [a_hi, a_lo] = split(a);
  const auto [b_hi, b_lo] = split(b);
  const T p = a * b;
  const T q = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
  return m - p <= q;
}

// Core: x is finite and positive.
template <typename T> constexpr T sqrt_core(T x) {
  auto [m, e] = binary_exponent(x);
  if (e % 2 != 0) {
    m *= T(2);
    --e;
  }
  const T scale = pow2<T>(e / 2);
  T y = T(0.5) * (m + T(1));
  for (;;) {
    const T next = T(0.5) * (y + m / y);
    if (!(next < y))
      break;
    y = next;
  }
  // Newton's rounding can leave y an ulp off; step to the nearest.
  for (;;) {
    const T down = sqrt_pred(y), up = sqrt_succ(y);
    if (exact_le(m, down, y))
      y = down;
    else if (!exact_le(m, y, up))
      y = up;
    else
      return y * scale;
  }
}

} // namespace detail

// constexpr sqrt for floating-point types.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T sqrt(T x) {
  return x != x      ? x                                   // NaN → NaN
         : x < T(0)  ? std::numeric_limits<T>::quiet_NaN() // negative → NaN
         : x == T(0) ? x                                   // ±0  → ±0
         : x == std::numeric_limits<T>::infinity() ? x     // +∞  → +∞
                                                   : detail::sqrt_core(x);
}

// Integral overload: promote to double.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr double sqrt(T x) {
  return cx::sqrt(static_cast<double>(x));
}

} // namespace cx

#endif // CX_STD_CX_SQRT_HPP
