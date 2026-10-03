// cx_trig.hpp — constexpr sin and cos via range reduction and Taylor series.
//
// Algorithm
// ---------
// 1. Range reduction: x = n·(π/2) + r with n = round(x / (π/2)), so
//    |r| ≤ π/4. π/2 is split into three 33-bit parts and a tail (the fdlibm
//    constants), so each n·part is exact for |n| < 2^20 and r keeps full
//    precision.
// 2. sin(r) and cos(r) by their Taylor series in Horner form:
//      sin(r) = r·(1 - r²/(2·3)·(1 - r²/(4·5)·(1 - …)))
//      cos(r) = 1 - r²/(1·2)·(1 - r²/(3·4)·(1 - …))
//    |r| ≤ π/4, so 12 terms give full double precision.
// 3. n mod 4 picks the quadrant: sin(x) is sin(r), cos(r), -sin(r) or
//    -cos(r); cos(x) is cos(r), -sin(r), -cos(r) or sin(r).
//
// Special cases: NaN → NaN, ±∞ → NaN, sin(±0) = ±0.
//
// Accuracy: within a few ulp of std::sin / std::cos for |x| < 1e6 (see
// tests/cx_std/cx_std.t.cpp). Beyond that the three-part reduction loses
// precision; there is no Payne–Hanek reduction.

#ifndef CX_STD_CX_TRIG_HPP
#define CX_STD_CX_TRIG_HPP

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cx {
namespace detail {

constexpr int kTrigTerms = 12;

constexpr double pio2_1 = 1.57079632673412561417e+00;
constexpr double pio2_2 = 6.07710050630396597660e-11;
constexpr double pio2_3 = 2.02226624871116645580e-21;
constexpr double pio2_3t = 8.47842766036889956997e-32;
constexpr double inv_pio2 = 6.36619772367581382433e-01;

// sin(r) and cos(r) for |r| ≤ π/4.
template <typename T> constexpr T sin_kernel(T r) {
  const T r2 = r * r;
  T s = T(1);
  for (int n = kTrigTerms; n >= 1; --n)
    s = T(1) - r2 / static_cast<T>((2 * n) * (2 * n + 1)) * s;
  return r * s;
}

template <typename T> constexpr T cos_kernel(T r) {
  const T r2 = r * r;
  T s = T(1);
  for (int n = kTrigTerms; n >= 1; --n)
    s = T(1) - r2 / static_cast<T>((2 * n - 1) * (2 * n)) * s;
  return s;
}

// Core: x is finite.
template <typename T> constexpr T trig_core(T x, bool want_cos) {
  const T q = x * static_cast<T>(inv_pio2);
  const long long n =
      static_cast<long long>(q < T(0) ? q - T(0.5) : q + T(0.5));
  const T nf = static_cast<T>(n);
  T r = x - nf * static_cast<T>(pio2_1);
  r -= nf * static_cast<T>(pio2_2);
  r -= nf * static_cast<T>(pio2_3);
  r -= nf * static_cast<T>(pio2_3t);
  const int quadrant = static_cast<int>(((n % 4) + 4) % 4) + (want_cos ? 1 : 0);
  switch (quadrant % 4) {
  case 0:
    return sin_kernel(r);
  case 1:
    return cos_kernel(r);
  case 2:
    return -sin_kernel(r);
  default:
    return -cos_kernel(r);
  }
}

} // namespace detail

// constexpr sin for floating-point types.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T sin(T x) {
  return x != x ? x // NaN → NaN
         : x == std::numeric_limits<T>::infinity() ||
                 x == -std::numeric_limits<T>::infinity()
             ? std::numeric_limits<T>::quiet_NaN() // ±∞ → NaN
         : x == T(0) ? x                           // ±0 → ±0
                     : detail::trig_core(x, false);
}

// constexpr cos for floating-point types.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T cos(T x) {
  return x != x ? x // NaN → NaN
         : x == std::numeric_limits<T>::infinity() ||
                 x == -std::numeric_limits<T>::infinity()
             ? std::numeric_limits<T>::quiet_NaN() // ±∞ → NaN
             : detail::trig_core(x, true);
}

// Integral overloads: promote to double.
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr double sin(T x) {
  return cx::sin(static_cast<double>(x));
}
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr double cos(T x) {
  return cx::cos(static_cast<double>(x));
}

} // namespace cx

#endif // CX_STD_CX_TRIG_HPP
