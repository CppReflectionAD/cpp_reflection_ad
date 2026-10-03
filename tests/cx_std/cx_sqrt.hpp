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
//
// Special cases: negative → NaN, +∞ → +∞, 0 → 0.
//
// Accuracy: within an ulp of std::sqrt (see tests/cx_std/cx_std.t.cpp).

#ifndef CX_STD_CX_SQRT_HPP
#define CX_STD_CX_SQRT_HPP

#include <limits>
#include <type_traits>

namespace cx {
namespace detail {

// Core: x is finite and positive.
template <typename T> constexpr T sqrt_core(T x) {
  constexpr T two_64 = T(18446744073709551616.0); // 2^64
  T m = x;
  T scale = T(1);
  while (m >= two_64 * two_64) {
    m /= two_64 * two_64;
    scale *= two_64;
  }
  while (m < T(1) / (two_64 * two_64)) {
    m *= two_64 * two_64;
    scale /= two_64;
  }
  while (m >= T(4)) {
    m /= T(4);
    scale *= T(2);
  }
  while (m < T(1)) {
    m *= T(4);
    scale /= T(2);
  }
  T y = T(0.5) * (m + T(1));
  for (;;) {
    const T next = T(0.5) * (y + m / y);
    if (!(next < y))
      break;
    y = next;
  }
  return y * scale;
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
