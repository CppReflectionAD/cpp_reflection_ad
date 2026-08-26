// cx_sqrt.hpp — constexpr sqrt via Newton-Raphson iteration.
//
// Algorithm
// ---------
// Newton-Raphson: given estimate xₙ, the next estimate is
//   xₙ₊₁ = 0.5 * (xₙ + x / xₙ)
// Iteration stops when the estimate stops changing (converged to machine
// precision), which happens in O(log(digits)) steps.
//
// Special cases: negative → NaN, +∞ → +∞, 0 → 0.

#ifndef CX_SQRT_HPP
#define CX_SQRT_HPP

#include <limits>
#include <type_traits>

namespace cx {
namespace detail {

template <typename T> constexpr T sqrt_iterate(T x, T curr, T prev) {
  return curr == prev ? curr
                      : sqrt_iterate(x, T(0.5) * (curr + x / curr), curr);
}

template <typename T> constexpr T sqrt_core(T x) {
  return sqrt_iterate(x, x, T(0));
}

} // namespace detail

// constexpr sqrt for floating-point types.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T sqrt(T x) {
  return x != x      ? x                                   // NaN → NaN
         : x < T(0)  ? std::numeric_limits<T>::quiet_NaN() // negative → NaN
         : x == T(0) ? T(0)                                //  0  → 0
         : x == std::numeric_limits<T>::infinity() ? x     // +∞  → +∞
                                                   : detail::sqrt_core(x);
}

// Integral overload: promote to double.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr double sqrt(T x) {
  return cx::sqrt(static_cast<double>(x));
}

} // namespace cx

#endif // CX_SQRT_HPP
