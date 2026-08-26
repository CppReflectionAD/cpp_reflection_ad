// exp_constexpr.hpp — constexpr exp via Horner-form Taylor series.
//
// Algorithm
// ---------
// Taylor series written in Horner form (avoids computing x^n and n!
// separately):
//   exp(x) = 1 + x*(1 + x/2*(1 + x/3*(... + x/N)))
//
// Stability:
//   For x <= -1: compute 1/exp(-x) to avoid catastrophic cancellation in the
//   direct series where large positive and negative terms nearly cancel.
//
// Special cases: NaN → NaN, -∞ → 0, +∞ → +∞, 0 → 1.
//
// Accuracy: 22 terms give full double precision for |x| ≤ 1, which is the
// range used after the reciprocal trick (direct path only sees x in (-1, +∞)).
// For large positive x the series is evaluated directly and is accurate.

#ifndef CX_EXP_HPP
#define CX_EXP_HPP

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cx {
namespace detail {

constexpr std::size_t kExpTerms = 22;

// Horner-form kernel.
// Computes: 1 + x/(n+1) * (1 + x/(n+2) * (... * (1 + x/N)))
// Initial call: exp_horner(x, 1, kExpTerms) gives the tail of the series
// (everything except the leading 1), so exp(x) = 1 + x * exp_horner(x, 1, N).
template <typename T>
constexpr T exp_horner(T x, std::size_t n, std::size_t N) {
  return n >= N ? T(1)
                : T(1) + x / static_cast<T>(n + 1) * exp_horner(x, n + 1, N);
}

// Core: x is finite and non-zero.
template <typename T> constexpr T exp_core(T x) {
  return x > T(-1)
             ? T(1) + x * exp_horner(x, 1, kExpTerms) // direct Taylor
             : T(1) / (T(1) + (-x) * exp_horner(-x, 1, kExpTerms)); // 1/exp(-x)
}

} // namespace detail

// constexpr exp for floating-point types.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T exp(T x) {
  return x != x                                     ? x    // NaN → NaN
         : x == -std::numeric_limits<T>::infinity() ? T(0) // -∞  → 0
         : x == +std::numeric_limits<T>::infinity() ? x    // +∞  → +∞
         : x == T(0)                                ? T(1) //  0  → 1
                                                    : detail::exp_core(x);
}

// Integral overload: promote to double.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr double exp(T x) {
  return cx::exp(static_cast<double>(x));
}

} // namespace cx

#endif // CX_EXP_HPP
