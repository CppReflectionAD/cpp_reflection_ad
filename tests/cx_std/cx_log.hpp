// cx_log.hpp — constexpr log via Taylor series with range reduction.
//
// Algorithm
// ---------
// 1. Range reduction: for x > √2, halve the argument recursively:
//      log(x) = 2 * log(√x)   until x ∈ (0, √2]
// 2. Flip for x < 1:
//      log(x) = -log(1/x)
// 3. Taylor series on the reduced range, u = x - 1 ∈ (-1, √2-1]:
//      log(1+u) = u * (1 - u/2 + u²/3 - u³/4 + …)  — Horner form
//    |u| ≤ √2-1 ≈ 0.414 gives rapid convergence; 30 terms yield full
//    double precision.
//
// Special cases: NaN → NaN, x<0 → NaN, 0 → -∞, +∞ → +∞, 1 → 0.

#ifndef CX_STD_CX_LOG_HPP
#define CX_STD_CX_LOG_HPP

#include "cx_sqrt.hpp"

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cx {
namespace detail {

constexpr std::size_t kLogTerms = 30;

// Horner-form kernel for log(1+u) / u.
// Computes: Σ_{n=0}^{N-1} (-u)^n / (n+1)  =  1 - u/2 + u²/3 - …
// so that log(1+u) = u * log_horner(u, 0, N).
template <typename T>
constexpr T log_horner(T u, std::size_t n, std::size_t N) {
  return n >= N ? T(0)
                : (n % 2 == 0 ? T(1) : T(-1)) / static_cast<T>(n + 1) +
                      u * log_horner(u, n + 1, N);
}

// Core: x is finite, positive, non-zero.
// Step 1 — range reduction to (0, √2].
// Step 2 — Taylor series on reduced range.
template <typename T> constexpr T log_core(T x) {
  return x > cx::sqrt(T(2)) // x > √2: halve via log(x) = 2*log(√x)
             ? T(2) * log_core(cx::sqrt(x))
             : x < T(1) // x < 1: flip via log(x) = -log(1/x)
                   ? -log_core(T(1) / x)
                   : (x - T(1)) * log_horner(x - T(1), 0, kLogTerms); // Taylor
}

} // namespace detail

// constexpr log for floating-point types.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T log(T x) {
  return x != x      ? x                                   // NaN  → NaN
         : x < T(0)  ? std::numeric_limits<T>::quiet_NaN() // x<0 → NaN
         : x == T(0) ? -std::numeric_limits<T>::infinity() //  0  → -∞
         : x == std::numeric_limits<T>::infinity() ? x     // +∞  → +∞
         : x == T(1)                               ? T(0)  //  1  → 0
                                                   : detail::log_core(x);
}

// Integral overload: promote to double.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr double log(T x) {
  return cx::log(static_cast<double>(x));
}

} // namespace cx

#endif // CX_STD_CX_LOG_HPP
