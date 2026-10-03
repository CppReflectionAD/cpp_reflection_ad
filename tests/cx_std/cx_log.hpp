// cx_log.hpp — constexpr log via binary range reduction and an atanh series.
//
// Algorithm
// ---------
// 1. Range reduction: x = m · 2^e with m ∈ [√½, √2). Scaling by powers of
//    two is exact.
// 2. log(m) = 2·atanh(u) with u = (m - 1) / (m + 1), |u| ≤ 0.172:
//      log(m) = 2·u·(1 + u²/3 + u⁴/5 + …)  — Horner form in u²
//    u² ≤ 0.03, so 14 terms give full double precision.
// 3. log(x) = e·ln2 + log(m), with ln2 split as in cx_exp.hpp so e·ln2_hi is
//    exact.
//
// Special cases: NaN → NaN, x<0 → NaN, 0 → -∞, +∞ → +∞, 1 → 0.
//
// Accuracy: within a few ulp of std::log (see tests/cx_std/cx_std.t.cpp).

#ifndef CX_STD_CX_LOG_HPP
#define CX_STD_CX_LOG_HPP

#include "cx_exp.hpp"

#include <cstddef>
#include <limits>
#include <numbers>
#include <type_traits>

namespace cx {
namespace detail {

constexpr std::size_t kLogTerms = 14;

// Horner-form kernel: Σ_{k=n}^{N-1} v^(k-n) / (2k+1), v = u².
template <typename T>
constexpr T log_horner(T v, std::size_t n, std::size_t N) {
  return n >= N
             ? T(0)
             : T(1) / static_cast<T>(2 * n + 1) + v * log_horner(v, n + 1, N);
}

// Core: x is finite, positive and not 1.
template <typename T> constexpr T log_core(T x) {
  constexpr T two_64 = T(18446744073709551616.0); // 2^64
  T m = x;
  int e = 0;
  while (m > two_64) {
    m /= two_64;
    e += 64;
  }
  while (m < T(1) / two_64) {
    m *= two_64;
    e -= 64;
  }
  while (m >= std::numbers::sqrt2_v<T>) {
    m /= T(2);
    ++e;
  }
  while (m < std::numbers::sqrt2_v<T> / T(2)) {
    m *= T(2);
    --e;
  }
  const T u = (m - T(1)) / (m + T(1));
  const T log_m = T(2) * u * log_horner(u * u, 0, kLogTerms);
  const T ef = static_cast<T>(e);
  return ef * static_cast<T>(ln2_hi) + (ef * static_cast<T>(ln2_lo) + log_m);
}

} // namespace detail

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T log(T x) {
  return x != x      ? x                                   // NaN  → NaN
         : x < T(0)  ? std::numeric_limits<T>::quiet_NaN() // x<0 → NaN
         : x == T(0) ? -std::numeric_limits<T>::infinity() //  0  → -∞
         : x == std::numeric_limits<T>::infinity() ? x     // +∞  → +∞
         : x == T(1)                               ? T(0)  //  1  → 0
                                                   : detail::log_core(x);
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr double log(T x) {
  return cx::log(static_cast<double>(x));
}

} // namespace cx

#endif // CX_STD_CX_LOG_HPP
