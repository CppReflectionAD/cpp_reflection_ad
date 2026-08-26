// cx_erfc.hpp — constexpr erfc via Taylor series (small |x|) and
// continued-fraction asymptotic expansion (large |x|).
//
// Algorithm
// ---------
// erfc(x) = 1 - erf(x), where
//   erf(x) = (2/√π) · Σ_{n=0}^{N} (-1)^n · x^(2n+1) / (n! · (2n+1))
//
// Written in Horner form for the variable t = -x²:
//   erf(x) = (2/√π) · x · Σ_{n=0}^{N} t^n / (n! · (2n+1))
//
// For |x| > 4 the Taylor series converges slowly; instead use the
// complementary asymptotic continued-fraction approximation:
//   erfc(x) ≈ (exp(-x²) / (x√π)) · cf(x²)
// via the Laplace continued fraction (truncated to kCFTerms levels).
//
// Special cases: NaN → NaN, +∞ → 0, -∞ → 2, 0 → 1.

#ifndef CX_ERFC_HPP
#define CX_ERFC_HPP

#include "cx_exp.hpp"

#include <cstddef>
#include <limits>
#include <numbers>
#include <type_traits>

namespace cx {
namespace detail {

constexpr std::size_t kErfTerms =
    28; // Taylor terms; covers double precision for |x| ≤ 4
constexpr std::size_t kCFTerms = 30; // continued-fraction levels for |x| > 4

// ---------------------------------------------------------------------------
// Taylor path: erf(x) = (2/√π)·x · Σ_{n=0}^{N} (-x²)^n / (n!·(2n+1))
// Horner kernel: computes Σ_{n=k}^{N} t^(n-k) / (n!·(2n+1))
// where t = -x², initial call n=0.
// ---------------------------------------------------------------------------
template <typename T>
constexpr T erf_horner(T t, T factorial_n, std::size_t n, std::size_t N) {
  return n >= N ? T(0)
                : T(1) / (factorial_n * static_cast<T>(2 * n + 1)) +
                      t * erf_horner(t, factorial_n * static_cast<T>(n + 1),
                                     n + 1, N);
}

template <typename T> constexpr T erfc_taylor(T x) {
  constexpr T two_over_sqrt_pi = T(2) * std::numbers::inv_sqrtpi_v<T>;
  T t = -x * x;
  T erf_x = two_over_sqrt_pi * x * erf_horner(t, T(1), 0, kErfTerms);
  return T(1) - erf_x;
}

// ---------------------------------------------------------------------------
// Asymptotic continued-fraction path for |x| > 4: erfc(x) via
// the Laplace CF: erfc(x) = (e^{-x²}/(x√π)) · 1/(1 + (1/2)/(x² + ...))
// Evaluates from the inside out (backwards recursion, N levels).
// CF tail value for level n: (n+0.5) / (x² + cf_tail(n+1))
// ---------------------------------------------------------------------------
template <typename T> constexpr T cf_tail(T x2, std::size_t n, std::size_t N) {
  return n >= N ? T(1)
                : T(1) + (static_cast<T>(n) + T(0.5)) /
                             (x2 * cf_tail(x2, n + 1, N));
}

template <typename T> constexpr T erfc_cf(T x) {
  constexpr T inv_sqrt_pi = std::numbers::inv_sqrtpi_v<T>;
  T x2 = x * x;
  // erfc(x) = exp(-x²) / (x · √π) · 1/cf_tail
  T cf = cf_tail(x2, 0, kCFTerms);
  return cx::exp(-x2) * inv_sqrt_pi / (x * cf);
}

// Core: x is finite.
template <typename T> constexpr T erfc_core(T x) {
  return (x > T(4) || x < T(-4)) ? (x > T(0) ? erfc_cf(x) : T(2) - erfc_cf(-x))
                                 : erfc_taylor(x);
}

} // namespace detail

// constexpr erfc for floating-point types.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T erfc(T x) {
  return x != x                                     ? x    // NaN → NaN
         : x == +std::numeric_limits<T>::infinity() ? T(0) // +∞  → 0
         : x == -std::numeric_limits<T>::infinity() ? T(2) // -∞  → 2
         : x == T(0)                                ? T(1) //  0  → 1
                                                    : detail::erfc_core(x);
}

// Integral overload: promote to double.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr double erfc(T x) {
  return cx::erfc(static_cast<double>(x));
}

} // namespace cx

#endif // CX_ERFC_HPP
