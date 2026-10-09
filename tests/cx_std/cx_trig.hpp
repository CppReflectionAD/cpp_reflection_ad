// cx_trig.hpp — constexpr sin and cos via range reduction and Taylor series.
//
// Algorithm
// ---------
// 1. Range reduction: x = n·(π/2) + r with n = round(x / (π/2)), so
//    |r| ≤ π/4.
//    - For |n| < 2^19, Cody–Waite: π/2 is split into three 33-bit parts and a
//      tail (the fdlibm constants), so each n·part is exact (up to 2^20) and
//      r keeps full precision.
//    - Beyond that, Payne–Hanek: x = m·2^e with m a 53-bit integer, and
//      x·(2/π) mod 4 is computed exactly in integers from a table of 2/π's
//      bits. Only the bits that affect it are used: earlier ones contribute
//      multiples of 4, and 7 words after them leave the fraction good to
//      about 2^-160, well past the closest any double comes to a multiple of
//      π/2 (about 2^-61).
// 2. sin(r) and cos(r) by their Taylor series in Horner form:
//      sin(r) = r·(1 - r²/(2·3)·(1 - r²/(4·5)·(1 - …)))
//      cos(r) = 1 - r²/(1·2)·(1 - r²/(3·4)·(1 - …))
//    |r| ≤ π/4, so 12 terms give full double precision.
// 3. n mod 4 picks the quadrant: sin(x) is sin(r), cos(r), -sin(r) or
//    -cos(r); cos(x) is cos(r), -sin(r), -cos(r) or sin(r).
//
// Everything is computed in double: float rounds the double result once, and
// long double is computed as double (NaN beyond the doubles' range).
//
// Special cases: NaN → NaN, ±∞ → NaN, sin(±0) = ±0.
//
// Accuracy: within a few ulp of std::sin / std::cos for every finite double
// (see tests/cx_std/cx_std.t.cpp).

#ifndef CX_STD_CX_TRIG_HPP
#define CX_STD_CX_TRIG_HPP

#include "cx_exp.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
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
// π/2 = pio2_hi + pio2_lo to about 2^-110
constexpr double pio2_hi = 1.57079632679489655800e+00;
constexpr double pio2_lo = 6.12323399573676603587e-17;

// 2/π's fraction, 32 bits a word: 2/π = Σ two_over_pi_words[j]·2^(-32(j+1)).
// 40 words reach past the last bit the largest double needs (the 37th).
inline constexpr std::uint32_t two_over_pi_words[] = {
    0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0, 0xdb629599, 0x3c439041,
    0xfe5163ab, 0xdebbc561, 0xb7246e3a, 0x424dd2e0, 0x06492eea, 0x09d1921c,
    0xfe1deb1c, 0xb129a73e, 0xe88235f5, 0x2ebb4484, 0xe99c7026, 0xb45f7e41,
    0x3991d639, 0x835339f4, 0x9c845f8b, 0xbdf9283b, 0x1ff897ff, 0xde05980f,
    0xef2f118b, 0x5a0a6d1f, 0x6d367ecf, 0x27cb09b7, 0x4f463f66, 0x9e5fea2d,
    0x7527bac7, 0xebe5f17b, 0x3d0739f7, 0x8a5292ea, 0x6bfb5fb1, 0x1f8d5d08,
    0x56033046, 0xfc7b6bab, 0xf0cfbc20, 0x9af4361d};

// sin(r) and cos(r) for |r| ≤ π/4.
constexpr double sin_kernel(double r) {
  const double r2 = r * r;
  double s = 1.0;
  for (int n = kTrigTerms; n >= 1; --n)
    s = 1.0 - r2 / static_cast<double>((2 * n) * (2 * n + 1)) * s;
  return r * s;
}

constexpr double cos_kernel(double r) {
  const double r2 = r * r;
  double s = 1.0;
  for (int n = kTrigTerms; n >= 1; --n)
    s = 1.0 - r2 / static_cast<double>((2 * n - 1) * (2 * n)) * s;
  return s;
}

// x = (4k + quadrant)·π/2 + r, |r| ≤ π/4 (up to rounding).
struct Reduced {
  int quadrant;
  double r;
};

// Cody–Waite, for |x·(2/π)| < 2^19.
constexpr Reduced reduce_small(double x, double q) {
  const long long n = static_cast<long long>(q < 0.0 ? q - 0.5 : q + 0.5);
  const double nf = static_cast<double>(n);
  double r = x - nf * pio2_1;
  r -= nf * pio2_2;
  r -= nf * pio2_3;
  r -= nf * pio2_3t;
  return {static_cast<int>(((n % 4) + 4) % 4), r};
}

// Payne–Hanek, for any finite |x| ≥ 2^19.
constexpr Reduced reduce_large(double x) {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
  const bool negative = (bits >> 63) != 0;
  // |x| = m·2^e; |x| ≥ 2^19 is normal, so m has its implicit bit.
  const std::uint64_t m =
      (bits & ((std::uint64_t{1} << 52) - 1)) | (std::uint64_t{1} << 52);
  const int e = static_cast<int>((bits >> 52) & 0x7ff) - 1075;

  // |x|·(2/π) = m·B·2^(e - 32(j0 + L)), B the words j0 .. j0+L-1 read as one
  // integer. A word j < j0 contributes a multiple of 2^(e - 32(j+1)) ≥ 4,
  // which changes neither the quadrant nor r.
  constexpr int L = 7;
  const int j0 = e >= 2 ? (e - 2) / 32 : 0;
  std::uint64_t b[L] = {}; // B, least significant word first
  for (int k = 0; k < L; ++k)
    b[k] = two_over_pi_words[j0 + L - 1 - k];

  // P = m·B in 32-bit limbs, least significant first: m = mh·2^32 + ml, and
  // each step's sum fits 64 bits ((2^32-1)^2 + 2·(2^32-1) < 2^64).
  std::uint64_t p[L + 2] = {};
  const std::uint64_t ml = m & 0xffffffff, mh = m >> 32;
  std::uint64_t carry = 0;
  for (int k = 0; k < L; ++k) {
    const std::uint64_t t = ml * b[k] + p[k] + carry;
    p[k] = t & 0xffffffff;
    carry = t >> 32;
  }
  p[L] = carry;
  carry = 0;
  for (int k = 0; k < L; ++k) {
    const std::uint64_t t = mh * b[k] + p[k + 1] + carry;
    p[k + 1] = t & 0xffffffff;
    carry = t >> 32;
  }
  p[L + 1] = carry;

  // P's bits [pos, pos + 64).
  const auto limb = [&](int i) { return i < L + 2 ? p[i] : std::uint64_t{0}; };
  const auto bits64 = [&](int pos) {
    const int w = pos / 32, s = pos % 32;
    std::uint64_t v = (limb(w) | limb(w + 1) << 32) >> s;
    if (s != 0)
      v |= limb(w + 2) << (64 - s);
    return v;
  };
  // P has `point` fraction bits (at least 32·L - 33): the integer part's
  // last two bits are the quadrant, and 128 bits below them the fraction.
  const int point = 32 * (j0 + L) - e;
  int quadrant = static_cast<int>(bits64(point) & 3);
  std::uint64_t hi = bits64(point - 64), lo = bits64(point - 128);
  // Round to the nearest n: a fraction of 1/2 or more is f - 1 from n + 1.
  bool r_negative = false;
  if ((hi >> 63) != 0) {
    quadrant = (quadrant + 1) & 3;
    r_negative = true;
    lo = ~lo + 1; // 2^128 - (hi, lo)
    hi = ~hi + (lo == 0 ? 1 : 0);
  }
  // |f| = (hi, lo)·2^-128, normalized so the top bit is set.
  int shift = 0;
  if (hi == 0) {
    hi = lo;
    lo = 0;
    shift = 64;
  }
  if (hi == 0)
    return {quadrant, 0.0}; // x is a multiple of π/2 to 2^-128
  const int z = std::countl_zero(hi);
  if (z != 0) {
    hi = hi << z | lo >> (64 - z);
    lo <<= z;
  }
  shift += z;
  // |f| = fh + fl: the top 53 bits exactly, then the rest.
  const double fh = static_cast<double>(hi >> 11) * pow2<double>(-53 - shift);
  const double fl =
      static_cast<double>(hi & 0x7ff) * pow2<double>(-64 - shift) +
      static_cast<double>(lo) * pow2<double>(-128 - shift);
  // r = f·π/2, with the products that matter in double-double.
  double r = fh * pio2_hi + (fl * pio2_hi + fh * pio2_lo);
  if (r_negative)
    r = -r;
  if (negative)
    return {(4 - quadrant) & 3, -r};
  return {quadrant, r};
}

// Core: x is finite.
constexpr double trig_core(double x, bool want_cos) {
  const double q = x * inv_pio2;
  const Reduced red =
      q > -0x1p19 && q < 0x1p19 ? reduce_small(x, q) : reduce_large(x);
  switch ((red.quadrant + (want_cos ? 1 : 0)) % 4) {
  case 0:
    return sin_kernel(red.r);
  case 1:
    return cos_kernel(red.r);
  case 2:
    return -sin_kernel(red.r);
  default:
    return -cos_kernel(red.r);
  }
}

// x as a double, if it is one: NaN for a long double beyond the doubles.
template <typename T> constexpr double as_double(T x) {
  if constexpr (sizeof(T) > sizeof(double)) {
    if (x > T(std::numeric_limits<double>::max()) ||
        x < T(-std::numeric_limits<double>::max()))
      return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(x);
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
         : detail::as_double(x) != detail::as_double(x)
             ? std::numeric_limits<T>::quiet_NaN()
             : static_cast<T>(detail::trig_core(detail::as_double(x), false));
}

// constexpr cos for floating-point types.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
constexpr T cos(T x) {
  return x != x ? x // NaN → NaN
         : x == std::numeric_limits<T>::infinity() ||
                 x == -std::numeric_limits<T>::infinity()
             ? std::numeric_limits<T>::quiet_NaN() // ±∞ → NaN
         : detail::as_double(x) != detail::as_double(x)
             ? std::numeric_limits<T>::quiet_NaN()
             : static_cast<T>(detail::trig_core(detail::as_double(x), true));
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
