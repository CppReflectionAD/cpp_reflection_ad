// cx_std accuracy: each constexpr function against <cmath>, in ulps, over the
// ranges the discontinuity analysis and limit algebra evaluate them on, plus
// the special values and the extremes constant evaluation must get through.

#include "cx_erfc.hpp"
#include "cx_exp.hpp"
#include "cx_log.hpp"
#include "cx_sqrt.hpp"
#include "cx_trig.hpp"
#include <test_simple_include.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr double inf = std::numeric_limits<double>::infinity();

// Distance between two doubles in units in the last place.
double ulps(double a, double b) {
  if (a == b)
    return 0.0;
  if (std::isnan(a) || std::isnan(b))
    return inf;
  const auto key = [](double d) {
    const std::int64_t i = std::bit_cast<std::int64_t>(d);
    return i < 0 ? std::numeric_limits<std::int64_t>::min() - i : i;
  };
  return std::fabs(static_cast<double>(key(a) - key(b)));
}

// Largest ulp error of cx_f against std_f over n random doubles of either
// sign, with exponents uniform over [2^min_exp, 2^1023].
template <class CxF, class StdF>
double max_ulps_any_exponent(CxF cx_f, StdF std_f, int min_exp, int n) {
  std::uint64_t state = 42; // xorshift64
  const auto next = [&] {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const std::uint64_t exponent =
        static_cast<std::uint64_t>(1023 + min_exp) +
        next() % static_cast<std::uint64_t>(1023 - min_exp + 1);
    const double x = std::bit_cast<double>(
        exponent << 52 | (next() & ((std::uint64_t{1} << 52) - 1)) |
        (next() & (std::uint64_t{1} << 63)));
    worst = std::fmax(worst, ulps(cx_f(x), std_f(x)));
  }
  return worst;
}

// Largest ulp error of cx_f against std_f over n points of [lo, hi], spaced
// evenly or (log_spaced) geometrically.
template <class CxF, class StdF>
double max_ulps(CxF cx_f, StdF std_f, double lo, double hi,
                bool log_spaced = false, int n = 20000) {
  double worst = 0.0;
  for (int i = 0; i <= n; ++i) {
    const double t = static_cast<double>(i) / n;
    const double x =
        log_spaced ? lo * std::pow(hi / lo, t) : lo + (hi - lo) * t;
    worst = std::fmax(worst, ulps(cx_f(x), std_f(x)));
  }
  return worst;
}

// Constant evaluation gets through the extremes: overflow, underflow,
// subnormals, and the depth of each series.
static_assert(cx::exp(1e5) == inf && cx::exp(-1e5) == 0.0);
static_assert(cx::exp(709.0) > 8e307 && cx::exp(-744.0) > 0.0);
// The largest double and float whose exp is finite, and the next ones up
constexpr double exp_max = 0x1.62e42fefa39efp+9; // 709.782712893383973...
static_assert(cx::exp(exp_max) > 1.79e308 && cx::exp(exp_max) < inf);
static_assert(cx::exp(0x1.62e42fefa39f0p+9) == inf);
static_assert(cx::exp(0x1.62e42ep+6f) > 3.4e38f &&
              cx::exp(0x1.62e430p+6f) ==
                  std::numeric_limits<float>::infinity());
// The smallest double and float whose exp is not 0, and the next ones down
// (std::exp's own boundaries; ln2_hi alone cut off 2e-7 above them)
constexpr double exp_min = -0x1.74910d52d3051p+9; // -745.133219101941108...
static_assert(cx::exp(exp_min) == 0x1p-1074 &&
              cx::exp(-0x1.74910d52d3052p+9) == 0.0);
static_assert(cx::exp(-0x1.9fe368p+6f) == 0x1p-149f &&
              cx::exp(-0x1.9fe36ap+6f) == 0.0f);
static_assert(cx::log(5e-324) < -744.0 && cx::log(1e308) > 709.0);
static_assert(cx::sqrt(5e-324) > 0.0 && cx::sqrt(1e308) > 1e153);
// Correctly rounded where Newton's iteration alone stops an ulp high
static_assert(cx::sqrt(8.9049066160978575e+299) == 9.4365812750687717e+149);
static_assert(cx::erfc(26.0) > 0.0 && cx::erfc(30.0) == 0.0);
static_assert(cx::erfc(0.5) > 0.47 && cx::erfc(-3.0) < 2.0);
static_assert(cx::sin(1e5) != 0.0 && cx::cos(-1e5) != 0.0);
// Beyond the Cody-Waite range, the reduction is exact (Payne-Hanek): the
// correctly rounded values (from mpmath), where the three-part reduction gave
// sin(1e18) = -2e29. The first is the double nearest a multiple of pi/2, at
// 2^-60.9.
constexpr double nearest_pio2_multiple = 6381956970095103.0 * 0x1p797;
static_assert(cx::sin(nearest_pio2_multiple) == 1.0 &&
              cx::cos(nearest_pio2_multiple) == -4.687165924254628e-19);
static_assert(cx::sin(1e18) == -0.9929693207404051 &&
              cx::cos(1e18) == 0.11837199021871073);
static_assert(cx::sin(1.7976931348623157e308) > 0.00496 &&
              cx::sin(1.7976931348623157e308) < 0.00497);

// Special values
static_assert(cx::exp(0.0) == 1.0 && cx::exp(-inf) == 0.0 &&
              cx::exp(inf) == inf);
static_assert(cx::log(1.0) == 0.0 && cx::log(0.0) == -inf &&
              cx::log(inf) == inf && cx::log(-1.0) != cx::log(-1.0));
static_assert(cx::sqrt(0.0) == 0.0 && cx::sqrt(inf) == inf &&
              cx::sqrt(-1.0) != cx::sqrt(-1.0));
static_assert(cx::erfc(0.0) == 1.0 && cx::erfc(inf) == 0.0 &&
              cx::erfc(-inf) == 2.0);
static_assert(cx::sin(0.0) == 0.0 && cx::cos(0.0) == 1.0 &&
              cx::sin(inf) != cx::sin(inf) && cx::cos(-inf) != cx::cos(-inf));

} // namespace

int main() {
  const auto cx_exp = [](double x) { return cx::exp(x); };
  const auto std_exp = [](double x) { return std::exp(x); };
  const auto cx_log = [](double x) { return cx::log(x); };
  const auto std_log = [](double x) { return std::log(x); };
  const auto cx_sqrt = [](double x) { return cx::sqrt(x); };
  const auto std_sqrt = [](double x) { return std::sqrt(x); };
  const auto cx_erfc = [](double x) { return cx::erfc(x); };
  const auto std_erfc = [](double x) { return std::erfc(x); };
  const auto cx_sin = [](double x) { return cx::sin(x); };
  const auto std_sin = [](double x) { return std::sin(x); };
  const auto cx_cos = [](double x) { return cx::cos(x); };
  const auto std_cos = [](double x) { return std::cos(x); };

  // exp: the whole range, including results that are subnormal
  EXPECT_LESS_THAN(max_ulps(cx_exp, std_exp, exp_min, exp_max), 4.0);
  // ... and densely up to overflow and underflow, where ln2_hi alone once cut
  // off 2e-7 short of them
  EXPECT_LESS_THAN(max_ulps(cx_exp, std_exp, exp_max - 3e-7, exp_max), 4.0);
  EXPECT_LESS_THAN(max_ulps(cx_exp, std_exp, exp_min, exp_min + 3e-7), 4.0);
  EXPECT_LESS_THAN(max_ulps(cx_exp, std_exp, -2.0, 2.0), 4.0);
  // log: normal and subnormal arguments, and around 1
  EXPECT_LESS_THAN(max_ulps(cx_log, std_log, 1e-308, 1e308, true), 4.0);
  EXPECT_LESS_THAN(max_ulps(cx_log, std_log, 5e-324, 1e-300, true), 4.0);
  EXPECT_LESS_THAN(max_ulps(cx_log, std_log, 0.5, 2.0), 4.0);
  // sqrt: correctly rounded, as IEEE requires of std::sqrt
  EXPECT_LESS_THAN(max_ulps(cx_sqrt, std_sqrt, 5e-324, 1e308, true), 0.5);
  EXPECT_LESS_THAN(max_ulps(cx_sqrt, std_sqrt, 1.0, 4.0, false, 1000000), 0.5);
  // erfc: the Taylor and continued-fraction paths, out to underflow
  EXPECT_LESS_THAN(max_ulps(cx_erfc, std_erfc, -6.0, 0.5), 8.0);
  EXPECT_LESS_THAN(max_ulps(cx_erfc, std_erfc, 0.5, 4.0), 8.0);
  EXPECT_LESS_THAN(max_ulps(cx_erfc, std_erfc, 4.0, 26.5), 8.0);
  // sin / cos: small arguments, |x| up to 1e5 ...
  EXPECT_LESS_THAN(max_ulps(cx_sin, std_sin, -10.0, 10.0), 4.0);
  EXPECT_LESS_THAN(max_ulps(cx_sin, std_sin, -1e5, 1e5), 4.0);
  EXPECT_LESS_THAN(max_ulps(cx_sin, std_sin, 1e-300, 1e-3, true), 4.0);
  EXPECT_LESS_THAN(max_ulps(cx_cos, std_cos, -10.0, 10.0), 4.0);
  EXPECT_LESS_THAN(max_ulps(cx_cos, std_cos, -1e5, 1e5), 4.0);
  // ... and every exponent, out to the largest double
  EXPECT_LESS_THAN(max_ulps_any_exponent(cx_sin, std_sin, -30, 400000), 4.0);
  EXPECT_LESS_THAN(max_ulps_any_exponent(cx_cos, std_cos, -30, 400000), 4.0);

  TEST_END;
}
