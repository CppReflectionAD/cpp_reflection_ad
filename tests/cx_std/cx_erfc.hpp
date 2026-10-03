// cx_erfc.hpp — constexpr erfc via a Taylor series (small x) and Laplace's
// continued fraction (larger x).
//
// Algorithm
// ---------
// x < 0: erfc(x) = 2 - erfc(-x), which loses nothing since erfc(-x) < 1.
//
// 0 < x < 0.5: erfc(x) = 1 - erf(x), with
//   erf(x) = (2/√π) · x · Σ_{n=0}^{N} t^n / (n! · (2n+1)),   t = -x²
// in Horner form. erf(x) < 0.53 here, so the subtraction loses little.
//
// x ≥ 0.5: erfc(x) = exp(-x²)/√π · K(x), where K is Laplace's continued
// fraction (Abramowitz & Stegun 7.1.14)
//   K(x) = 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + …)))))
// evaluated from the inside out. It converges for every x > 0, faster as x
// grows, so the depth is chosen from x. exp(-x²) amplifies the rounding of
// x² by x², so x² is split exactly into hi + lo first (Veltkamp/Dekker).
//
// Special cases: NaN → NaN, +∞ → 0, -∞ → 2, 0 → 1.
//
// Accuracy: within a few ulp of std::erfc (see tests/cx_std/cx_std.t.cpp).

#ifndef CX_STD_CX_ERFC_HPP
#define CX_STD_CX_ERFC_HPP

#include "cx_exp.hpp"

#include <cstddef>
#include <limits>
#include <numbers>
#include <type_traits>

namespace cx {
namespace detail {

constexpr std::size_t kErfTerms = 28; // Taylor terms; ample for |x| < 0.5

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
// Continued-fraction path, x ≥ 0.5.
// ---------------------------------------------------------------------------

// Levels of K(x) for full precision: the truncation error shrinks roughly
// like exp(-4x·√(levels/2)).
template <typename T> constexpr std::size_t erfc_cf_levels(T x) {
  return static_cast<std::size_t>(T(40) + T(700) / (x * x));
}

// exp(-x²), with x² = hi + lo exactly.
template <typename T> constexpr T exp_minus_square(T x) {
  constexpr T splitter =
      pow2<T>((std::numeric_limits<T>::digits + 1) / 2) + T(1);
  const T c = splitter * x;
  const T hi = c - (c - x);
  const T lo = x - hi;
  return cx::exp(-(hi * hi)) * cx::exp(-(T(2) * hi * lo + lo * lo));
}

template <typename T> constexpr T erfc_cf(T x) {
  T k = T(0); // K's tail, from the inside out
  for (std::size_t n = erfc_cf_levels(x); n >= 1; --n)
    k = (static_cast<T>(n) / T(2)) / (x + k);
  return exp_minus_square(x) * std::numbers::inv_sqrtpi_v<T> / (x + k);
}

// Core: x is finite and non-zero.
template <typename T> constexpr T erfc_core(T x) {
  if (x < T(0))
    return T(2) - erfc_core(-x);
  if (x > T(30)) // underflows: erfc(27.3) is below the smallest double
    return T(0);
  return x < T(0.5) ? erfc_taylor(x) : erfc_cf(x);
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

#endif // CX_STD_CX_ERFC_HPP
