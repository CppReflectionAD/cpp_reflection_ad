// exp_constexpr.hpp — constexpr exp via range reduction and a Taylor series.
//
// Algorithm
// ---------
// 1. Range reduction: x = k·ln2 + r with k = round(x / ln2), so |r| ≤ ln2/2.
//    ln2 is split into a high part with trailing zero bits and a low
//    correction (Cody–Waite), so k·ln2_hi is exact and r keeps full precision.
// 2. exp(r) by its Taylor series in Horner form (avoids computing r^n and n!
//    separately):
//      exp(r) = 1 + r*(1 + r/2*(1 + r/3*(... + r/N)))
//    22 terms are far more than |r| ≤ 0.35 needs for double precision.
// 3. exp(x) = exp(r) · 2^k. Scaling by a power of two is exact; 2^k is
//    applied in two halves so neither overflows nor underflows before the
//    result does.
//
// Special cases: NaN → NaN, -∞ → 0, +∞ → +∞, 0 → 1. Beyond the type's range
// the result is +∞ or 0, as std::exp.
//
// Accuracy: within a few ulp of std::exp over the whole range (see
// tests/cx_std/cx_std.t.cpp).

#ifndef CX_STD_CX_EXP_HPP
#define CX_STD_CX_EXP_HPP

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cx {
namespace detail {

constexpr std::size_t kExpTerms = 22;

// ln2 = ln2_hi + ln2_lo, where ln2_hi has 32 significant bits so k·ln2_hi is
// exact for |k| < 2^21 (the fdlibm split).
constexpr double ln2_hi = 6.93147180369123816490e-01;
constexpr double ln2_lo = 1.90821492927058770002e-10;
constexpr double inv_ln2 = 1.44269504088896338700e+00;

// Horner-form kernel.
// Computes: 1 + x/(n+1) * (1 + x/(n+2) * (... * (1 + x/N)))
// Initial call: exp_horner(x, 1, kExpTerms) gives the tail of the series
// (everything except the leading 1), so exp(x) = 1 + x * exp_horner(x, 1, N).
template <typename T>
constexpr T exp_horner(T x, std::size_t n, std::size_t N) {
  return n >= N ? T(1)
                : T(1) + x / static_cast<T>(n + 1) * exp_horner(x, n + 1, N);
}

// 2^k, exactly, by repeated squaring; 2^k must be within the type's range.
// The base is squared only while bits remain, so no step overflows (GCC
// rejects an overflow during constant evaluation).
template <typename T> constexpr T pow2(int k) {
  T base = k < 0 ? T(0.5) : T(2);
  unsigned n = static_cast<unsigned>(k < 0 ? -k : k);
  T result = T(1);
  for (; n != 0; n >>= 1) {
    if (n & 1u)
      result *= base;
    if (n > 1u)
      base *= base;
  }
  return result;
}

// v = hi + lo exactly, each half of T's significand (Veltkamp's split), so
// products of halves are exact: the core of an error-free product (Dekker).
// |v| must be well within range: v · 2^(digits/2) must not overflow.
template <typename T> struct Halves {
  T hi, lo;
};
template <typename T> constexpr Halves<T> split(T v) {
  constexpr T splitter =
      pow2<T>((std::numeric_limits<T>::digits + 1) / 2) + T(1);
  const T t = splitter * v;
  const T hi = t - (t - v);
  return {hi, v - hi};
}

// x = m · 2^e with m ∈ [1, 2), for finite positive x (frexp, scaled by 2).
// Scaling by powers of two is exact. The coarse steps are 2^64, within even
// float's range: a larger one overflows there, which GCC rejects during
// constant evaluation.
template <typename T> struct BinaryExponent {
  T m;
  int e;
};
template <typename T> constexpr BinaryExponent<T> binary_exponent(T x) {
  constexpr T two_64 = pow2<T>(64);
  T m = x;
  int e = 0;
  while (m >= two_64) {
    m /= two_64;
    e += 64;
  }
  while (m < T(1) / two_64) {
    m *= two_64;
    e -= 64;
  }
  while (m >= T(2)) {
    m /= T(2);
    ++e;
  }
  while (m < T(1)) {
    m *= T(2);
    --e;
  }
  return {m, e};
}

// Nearest integer to x, |x| small enough to fit an int.
template <typename T> constexpr int round_to_int(T x) {
  return static_cast<int>(x < T(0) ? x - T(0.5) : x + T(0.5));
}

// Core: x is finite and non-zero.
template <typename T> constexpr T exp_core(T x) {
  using limits = std::numeric_limits<T>;
  // ln2_hi has more bits than float's significand, so k·ln2_hi is not exact
  // in float. Work in double and round once. Converting a double past T's
  // rounding threshold to T is undefined, so that is +∞ here.
  if constexpr (limits::digits < std::numeric_limits<double>::digits) {
    const double y = exp_core(static_cast<double>(x));
    if (y >= pow2<double>(limits::max_exponent) -
                 pow2<double>(limits::max_exponent - limits::digits - 1))
      return limits::infinity();
    return static_cast<T>(y);
  }
  // Past the type's range the result is +∞ or 0. Return those directly:
  // computing them by overflow is not a constant expression for GCC.
  // Above max_exponent·ln2 e^x is past the largest finite value; below
  // (min_exponent - digits - 1)·ln2 it is under half the smallest subnormal.
  // Both use the full ln2: ln2_hi alone is 2e-7 short of it at double.
  const T max_exponent = static_cast<T>(limits::max_exponent);
  const T min_exponent =
      static_cast<T>(limits::min_exponent - limits::digits - 1);
  if (x > max_exponent * static_cast<T>(ln2_hi) +
              max_exponent * static_cast<T>(ln2_lo))
    return limits::infinity();
  if (x < min_exponent * static_cast<T>(ln2_hi) +
              min_exponent * static_cast<T>(ln2_lo))
    return T(0);
  const int k = round_to_int(x * static_cast<T>(inv_ln2));
  const T r = (x - static_cast<T>(k) * static_cast<T>(ln2_hi)) -
              static_cast<T>(k) * static_cast<T>(ln2_lo);
  const T exp_r = T(1) + r * exp_horner(r, 1, kExpTerms);
  // So k ≤ max_exponent, and exp_r·2^k is finite exactly when exp_r < 1
  // there (scaling by a power of two is exact).
  if (k == limits::max_exponent && exp_r >= T(1))
    return limits::infinity();
  return exp_r * pow2<T>(k / 2) * pow2<T>(k - k / 2);
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

#endif // CX_STD_CX_EXP_HPP
