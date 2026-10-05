#ifndef DISCONTINUITY_ANALYSIS_HPP
#define DISCONTINUITY_ANALYSIS_HPP

// discontinuity_analysis.hpp — compile-time discontinuity point extraction.
//
// Given a function f(x₀, x₁, ..., xₙ) and a target input index i, with fixed
// values for all other inputs, finds all points where f jumps along the i-th
// argument, and by how much.
//
// Usage:
//   double f(double x, double y) { return (x > y) ? 1.0 : 0.0; }
//   constexpr auto points = ad::get_discontinuity_points<^^f, 0>(2.0);
//   // points = [2.0] (discontinuity at x = y = 2.0)
//
// The fixed values are the other arguments in order, the target's skipped:
// for g(k, s) with target 1, pass k.
//
// Strategy:
//   1. Build the DAG via reflection (like autograd.h)
//   2. Find each crossing: a comparison with a side that varies continuously
//      with the target, or such a value used directly as a condition
//      (`s ? a : b`, `!(s - k)`, which test v != 0). Solve lhs - rhs = 0 (or
//      v = 0) for the target. Both sides must be affine in it (built with
//      + -, unary - and * / by target-free values, and `?:` on target-free
//      conditions); anything else is a compile-time error rather than a
//      guess. A comparison of values that change only in steps, e.g.
//      `(s > k ? 1 : 0) > 0.5`, needs no root of its own: it flips only where
//      a comparison inside it does.
//   3. Roots a few ulps apart or less are one point: often one crossing
//      written two ways, whose roots round apart. (So two real jumps that
//      close are reported as one, their sum.) At each point, evaluate the
//      whole function with the crossings rooted there forced to their outcome
//      just above, and just below, it. The difference is the jump. Points
//      with no jump are dropped, and so are those whose jump is only
//      rounding, as at a kink whose root is not a double: the function is
//      also evaluated with error bounds (midpoint-radius), over every target
//      the exact roots could be at, and the jump must exceed them.
//   4. Return as a static array (assuming finite discontinuities)
//
// get_discontinuity_points_and_amplitudes also gives each point's amplitude:
// the jump f(x+) - f(x-) as the target increases through the point x. A
// digital call (s > k ? 1 : 0) jumps by +1 at k, a digital put
// (s < k ? 1 : 0) by -1. It is not "true branch minus false branch", which
// has the wrong sign wherever the comparison holds below the point.
//
// Everything is evaluated as Fn evaluates it: branches it does not take are
// skipped, and arithmetic follows IEEE (1 / 0 is inf, NaN compares false)
// even during constant evaluation, where such operations would otherwise
// stop the build. exp/log/sqrt/erfc/sin/cos go through cx_std in every entry
// point, so the consteval and runtime (_rt) entry points give identical
// results; these may differ from Fn's own <cmath> results in the last bits.
//
// It is an error (a compile error from the consteval entry points, an
// exception from _rt) for Fn not to be finite on either side of a crossing
// point -- a pole such as `s > 0 ? k / s : 0` is not a jump -- for a
// crossing's slope in the target not to be finite at the fixed values (`s / k
// > 1` at k = 0, which does jump at s = 0, but also `s * (k - k) > 0` at
// k = inf, which never flips), for its sides to overflow with the target at
// 0 (`(s - k) * 1e10 > 0` at k = 1e300), for a jump's rounding error not to
// be bounded, or to have more points than MaxPoints.

#include "../autograd.h"
#include "../cx_std/cx_erfc.hpp"
#include "../cx_std/cx_exp.hpp"
#include "../cx_std/cx_log.hpp"
#include "../cx_std/cx_sqrt.hpp"
#include "../cx_std/cx_trig.hpp"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <meta>
#include <utility>
#include <vector>

namespace ad {

using std::meta::info;

// ---------------------------------------------------------------------------
// Main API: get_discontinuity_points<Fn, TargetArgIndex>(fixed_args...)
//
// Returns a static array of discontinuity points along TargetArgIndex
// while the other arguments are fixed at the provided values.
// ---------------------------------------------------------------------------

// Result type for discontinuity point extraction
template <std::size_t MaxPoints = 16> struct DiscontinuityPoints {
  std::array<double, MaxPoints> points = {};
  std::size_t count = 0;

  // Check if array is empty (no discontinuities found)
  constexpr bool empty() const { return count == 0; }

  // Get the number of valid discontinuity points
  constexpr std::size_t size() const { return count; }

  // Access individual points
  constexpr double operator[](std::size_t i) const {
    return (i < count) ? points[i] : 0.0;
  }

  // Iterator support for range-based loops
  constexpr auto begin() { return points.begin(); }
  constexpr auto end() { return points.begin() + count; }
  constexpr auto begin() const { return points.begin(); }
  constexpr auto end() const { return points.begin() + count; }
};

// Result type for discontinuity points with amplitudes
// Stores pairs: (point, amplitude), flattened into single array. The amplitude
// is the jump f(x+) - f(x-) as the target increases through the point x.
// E.g., points at x=1.0 with amplitude 1.0 and x=4.0 with amplitude -1.0
// would be stored as [1.0, 1.0, 4.0, -1.0]
template <std::size_t MaxPoints = 16> struct DiscontinuityPointsWithAmplitudes {
  std::array<double, MaxPoints * 2> data = {}; // pairs of (point, amplitude)
  std::size_t count = 0; // number of (point, amplitude) pairs

  // Check if array is empty (no discontinuities found)
  constexpr bool empty() const { return count == 0; }

  // Get the number of valid discontinuity pairs
  constexpr std::size_t size() const { return count; }

  // Get point at index i
  constexpr double point(std::size_t i) const {
    return (i < count) ? data[i * 2] : 0.0;
  }

  // Get amplitude at index i
  constexpr double amplitude(std::size_t i) const {
    return (i < count) ? data[i * 2 + 1] : 0.0;
  }

  // Access as flat array [point0, amp0, point1, amp1, ...]
  constexpr double operator[](std::size_t i) const { return data[i]; }

  // Appends a pair; the analysis adds them in order of their points. A point
  // that does not fit is an error, not a silent omission.
  constexpr void add_point_with_amplitude(double point, double amplitude) {
    if (count == MaxPoints)
      throw "discontinuity_analysis: Fn has more discontinuities than "
            "MaxPoints; pass a larger MaxPoints";
    data[count * 2] = point;
    data[count * 2 + 1] = amplitude;
    count++;
  }
};

// ---------------------------------------------------------------------------
// The analysis. Every entry point below calls detail_disc::analyze, so a fix
// to how comparisons, points or amplitudes are read lands in one place.
// ---------------------------------------------------------------------------
namespace detail_disc {

constexpr bool is_comparison(OpKind op) {
  return op == OpKind::Lt || op == OpKind::Le || op == OpKind::Gt ||
         op == OpKind::Ge || op == OpKind::Eq || op == OpKind::Ne;
}

// The comparisons whose outcome flips as the target crosses a point. `==` and
// `!=` are left out: they take their other outcome only at the point itself,
// so the function's left and right limits there are equal and there is no
// jump (a single point carries no probability mass; see #67).
constexpr bool is_ordering(OpKind op) {
  return op == OpKind::Lt || op == OpKind::Le || op == OpKind::Gt ||
         op == OpKind::Ge;
}

// The function's arguments: the fixed values in order, skipping the target's
// slot, which is left 0 (the analysis sets the target itself).
template <std::size_t Target, typename... FixedArgs>
constexpr std::array<double, sizeof...(FixedArgs) + 1>
make_inputs(FixedArgs... fixed_args) {
  const std::array<double, sizeof...(FixedArgs)> fixed = {
      static_cast<double>(fixed_args)...};
  std::array<double, sizeof...(FixedArgs) + 1> in = {};
  for (std::size_t i = 0; i < fixed.size(); ++i)
    in[i < Target ? i : i + 1] = fixed[i];
  return in;
}

// Which nodes are read as conditions: a Select's condition, the operands of
// `!`, `&&` and `||`, and every guard.
template <info Fn> consteval std::vector<char> read_as_condition() {
  const auto nodes = nodes_of<Fn>;
  std::vector<char> read(nodes.size(), 0);
  for (const Node &n : nodes) {
    if (n.op == OpKind::Select)
      read[n.cond] = 1;
    if (n.op == OpKind::Not || n.op == OpKind::And || n.op == OpKind::Or)
      read[n.a] = 1;
    if (n.op == OpKind::And || n.op == OpKind::Or)
      read[n.b] = 1;
    if (n.guard != UNGUARDED)
      read[n.guard] = 1;
  }
  return read;
}

// The crossings: the nodes whose truth flips at a point of their own as the
// target moves, which therefore have to be solved for and, at a shared point,
// held at their limits.
//  - A comparison with a side that varies continuously.
//  - A value that varies continuously and is used directly as a condition: it
//    tests v != 0. Like `!=` it never makes a point itself, but where it
//    shares one it must be held at its off-point truth.
template <info Fn, std::size_t Target> consteval std::vector<char> crossings() {
  const auto nodes = nodes_of<Fn>;
  const auto dep = target_dependence_of<Fn, Target>;
  const std::vector<char> read = read_as_condition<Fn>();
  std::vector<char> crossing(nodes.size(), 0);
  for (const Node &n : nodes) {
    if (is_comparison(n.op))
      crossing[n.self] =
          varies_continuously(dep[n.a]) || varies_continuously(dep[n.b]);
    else
      crossing[n.self] = read[n.self] && varies_continuously(dep[n.self]);
  }
  return crossing;
}

template <info Fn, std::size_t Target>
inline constexpr auto crossings_of =
    std::define_static_array(crossings<Fn, Target>());

// The crossings that can make a point: the ordering comparisons, in order.
template <info Fn, std::size_t Target>
consteval std::vector<std::size_t> ordering_crossings() {
  const auto nodes = nodes_of<Fn>;
  const auto crossing = crossings_of<Fn, Target>;
  std::vector<std::size_t> found;
  for (const Node &n : nodes)
    if (crossing[n.self] && is_ordering(n.op))
      found.push_back(n.self);
  return found;
}

template <info Fn, std::size_t Target>
inline constexpr auto ordering_crossings_of =
    std::define_static_array(ordering_crossings<Fn, Target>());

// What has to be evaluated to place every crossing: the crossings, the nodes
// they read, directly or not, and the guards that say whether those are
// reached at all.
template <info Fn, std::size_t Target>
consteval std::vector<char> crossing_cone() {
  const auto nodes = nodes_of<Fn>;
  const auto crossing = crossings_of<Fn, Target>;
  std::vector<char> cone(nodes.size(), 0);
  for (std::size_t i = nodes.size(); i-- > 0;) {
    const Node &n = nodes[i];
    if (!cone[i] && !crossing[i])
      continue;
    cone[i] = 1;
    if (op_has_a(n.op))
      cone[n.a] = 1;
    if (op_has_b(n.op))
      cone[n.b] = 1;
    if (op_has_cond(n.op))
      cone[n.cond] = 1;
    if (n.guard != UNGUARDED)
      cone[n.guard] = 1;
  }
  return cone;
}

template <info Fn, std::size_t Target>
inline constexpr auto crossing_cone_of =
    std::define_static_array(crossing_cone<Fn, Target>());

// Which nodes crossing i's gap reads, directly or not: its sides (or the
// condition itself) and their operands. Not their guards: whether a node is
// reached does not depend on the target, so any sweep already says.
template <info Fn> consteval std::vector<char> gap_reads(std::size_t i) {
  const auto nodes = nodes_of<Fn>;
  std::vector<char> read(nodes.size(), 0);
  if (is_comparison(nodes[i].op))
    read[nodes[i].a] = read[nodes[i].b] = 1;
  else
    read[i] = 1;
  for (std::size_t j = i + 1; j-- > 0;) {
    const Node &n = nodes[j];
    if (!read[j])
      continue;
    if (op_has_a(n.op))
      read[n.a] = 1;
    if (op_has_b(n.op))
      read[n.b] = 1;
    if (op_has_cond(n.op))
      read[n.cond] = 1;
  }
  return read;
}

// The nodes whose target derivative some crossing's gap reads: the union of
// their gap_reads. Only these need a tangent; a guard, or anything else in
// the crossing cone, needs just its value.
template <info Fn, std::size_t Target>
consteval std::vector<char> tangent_needed() {
  const auto crossing = crossings_of<Fn, Target>;
  std::vector<char> needed(crossing.size(), 0);
  for (std::size_t i = 0; i < crossing.size(); ++i) {
    if (!crossing[i])
      continue;
    const std::vector<char> read = gap_reads<Fn>(i);
    for (std::size_t j = 0; j < read.size(); ++j)
      needed[j] = needed[j] || read[j];
  }
  return needed;
}

template <info Fn, std::size_t Target>
inline constexpr auto tangent_needed_of =
    std::define_static_array(tangent_needed<Fn, Target>());

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

constexpr bool is_finite(double x) { return x == x && x != kInf && x != -kInf; }

// x moved k ulps up (k > 0) or down (k < 0), through ±0, stopping at ±inf.
constexpr double step_ulps(double x, int k) {
  const auto up = [](double v) {
    if (v == 0.0)
      return std::numeric_limits<double>::denorm_min();
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(v);
    return std::bit_cast<double>(v > 0.0 ? bits + 1 : bits - 1);
  };
  for (; k > 0 && is_finite(x); --k)
    x = up(x);
  for (; k < 0 && is_finite(x); ++k)
    x = -up(-x);
  return x;
}

// primal()'s arithmetic here: IEEE results even during constant evaluation,
// where an operation that divides by zero or produces a NaN otherwise stops
// the build. A NaN or infinity is produced directly instead, so a target-free
// value Fn never uses at these arguments (`k != 0 ? (s > 1 / k ? ...)` at
// k = 0) evaluates harmlessly, and a value Fn does use is the one Fn sees.
// The functions are cx_std's in every entry point, so the consteval and
// runtime entry points agree to the bit.
struct IeeeCxMath {
  static constexpr bool is_nan(double x) { return x != x; }
  static constexpr bool is_inf(double x) { return x == kInf || x == -kInf; }
  static constexpr bool sign_bit(double x) {
    return (std::bit_cast<std::uint64_t>(x) >> 63) != 0;
  }

  [[gnu::always_inline]] static constexpr double add(double a, double b) {
    if (is_nan(a) || is_nan(b) || (is_inf(a) && is_inf(b) && a != b))
      return kNaN;
    return a + b;
  }
  [[gnu::always_inline]] static constexpr double sub(double a, double b) {
    if (is_nan(a) || is_nan(b) || (is_inf(a) && is_inf(b) && a == b))
      return kNaN;
    return a - b;
  }
  [[gnu::always_inline]] static constexpr double mul(double a, double b) {
    if (is_nan(a) || is_nan(b) || (is_inf(a) && b == 0.0) ||
        (a == 0.0 && is_inf(b)))
      return kNaN;
    return a * b;
  }
  [[gnu::always_inline]] static constexpr double div(double a, double b) {
    if (is_nan(a) || is_nan(b) || (a == 0.0 && b == 0.0) ||
        (is_inf(a) && is_inf(b)))
      return kNaN;
    if (b == 0.0)
      return sign_bit(a) != sign_bit(b) ? -kInf : kInf;
    return a / b;
  }
  [[gnu::always_inline]] static constexpr double neg(double a) { return -a; }
  [[gnu::always_inline]] static constexpr double max(double a, double b) {
    return (a < b) ? b : a;
  }
  [[gnu::always_inline]] static constexpr double min(double a, double b) {
    return (b < a) ? b : a;
  }

  template <OpKind Op>
  [[gnu::always_inline]] static constexpr double unary(double x) {
    if constexpr (Op == OpKind::Exp)
      return cx::exp(x);
    else if constexpr (Op == OpKind::Log)
      return cx::log(x);
    else if constexpr (Op == OpKind::Sqrt)
      return cx::sqrt(x);
    else if constexpr (Op == OpKind::Erfc)
      return cx::erfc(x);
    else if constexpr (Op == OpKind::Sin)
      return cx::sin(x);
    else
      return cx::cos(x);
  }
};

// A value with a bound on its error, for telling a jump from rounding (see
// analyze). `mid` is the value exactly as IeeeCxMath computes it; the value
// in exact arithmetic -- the arguments as given, the target anywhere within
// its own rad of its mid -- is within `rad` of `mid`. An infinite mid is an
// overflow: the exact value has mid's sign and a magnitude of at least the
// largest double less `rad`, infinity included (so an input or constant that
// is itself ±inf is {±inf, 0}). An infinite rad, as for a NaN mid, bounds
// nothing. Comparisons read `mid`, as Fn's do its value.
struct Ball {
  double mid = 0.0;
  double rad = 0.0;

  friend constexpr bool operator<(Ball a, Ball b) { return a.mid < b.mid; }
  friend constexpr bool operator<=(Ball a, Ball b) { return a.mid <= b.mid; }
  friend constexpr bool operator>(Ball a, Ball b) { return a.mid > b.mid; }
  friend constexpr bool operator>=(Ball a, Ball b) { return a.mid >= b.mid; }
  friend constexpr bool operator==(Ball a, Ball b) { return a.mid == b.mid; }
  friend constexpr bool operator!=(Ball a, Ball b) { return a.mid != b.mid; }
};

constexpr double mid_of(double x) { return x; }
constexpr double mid_of(Ball x) { return x.mid; }

// primal()'s arithmetic on Balls: the mids as IeeeCxMath computes them, the
// radii by the usual first-principles bounds (midpoint-radius interval
// arithmetic), and an overflow's by the least magnitude its exact value can
// have, so that what Fn computes from it (1 / (1 + exp(1000))) stays bounded.
// The radii are computed only from finite values, so they can overflow to inf
// but never become NaN, which would stop the build.
struct BallCxMath {
  using M = IeeeCxMath;
  // One rounding: at most half an ulp, |r|·2^-53, plus the smallest
  // subnormal where a product or quotient underflows.
  static constexpr double kHalfUlp = 0x1p-53;
  static constexpr double kTiny = std::numeric_limits<double>::denorm_min();
  static constexpr double kMax = std::numeric_limits<double>::max();
  // cx_std's error, in the same units: 8 ulps, against a measured worst of 5
  // (erfc).
  static constexpr double kCxError = 16.0;

  static constexpr double abs(double x) { return x < 0.0 ? -x : x; }
  static constexpr double max(double a, double b) { return a < b ? b : a; }
  static constexpr double min(double a, double b) { return b < a ? b : a; }
  // No bound: a NaN, or an operand with none.
  static constexpr bool bounded(Ball a) {
    return a.mid == a.mid && a.rad != kInf;
  }
  static constexpr bool unbounded(double mid, Ball a, Ball b = {}) {
    return mid != mid || !bounded(a) || !bounded(b);
  }
  // cx_std's error at a result r. (16 · 2^-53 is a power of two, so this
  // rounds as 16 · |r| · 2^-53 would, without overflowing past 1e307.)
  static constexpr double cx_error(double r) {
    return abs(r) * (kCxError * kHalfUlp) + kTiny;
  }
  // The least magnitude a's exact value can have, or 0.
  static constexpr double least_of(Ball a) {
    const double least = is_finite(a.mid) ? abs(a.mid) - a.rad : kMax - a.rad;
    return least > 0.0 ? least : 0.0;
  }
  // The most: a's reach from 0, for a finite mid.
  static constexpr double most_of(Ball a) { return abs(a.mid) + a.rad; }
  // An overflow to mid = ±inf whose exact value is at least `least` in
  // magnitude: no bound unless that is more than 0, which fixes the sign.
  static constexpr Ball overflow(double mid, double least) {
    if (!(least > 0.0))
      return {mid, kInf};
    return {mid, least < kMax ? kMax - least : 0.0};
  }

  [[gnu::always_inline]] static constexpr Ball add(Ball a, Ball b) {
    const double mid = M::add(a.mid, b.mid);
    if (unbounded(mid, a, b))
      return {mid, kInf};
    if (!is_finite(mid)) {
      // Finite mids: their exact sum is past the largest double, and the
      // radii move it by at most their sum. Past an infinite operand (of
      // mid's sign), a finite one moves it by at most its reach, and another
      // infinite one only further out.
      const bool a_finite = is_finite(a.mid), b_finite = is_finite(b.mid);
      return overflow(mid, a_finite && b_finite ? kMax - (a.rad + b.rad)
                           : a_finite           ? least_of(b) - most_of(a)
                           : b_finite           ? least_of(a) - most_of(b)
                                      : max(least_of(a), least_of(b)));
    }
    // A sum that rounds to 0, or to a subnormal, is exact.
    return {mid, a.rad + b.rad + abs(mid) * kHalfUlp};
  }
  [[gnu::always_inline]] static constexpr Ball sub(Ball a, Ball b) {
    return add(a, neg(b));
  }
  [[gnu::always_inline]] static constexpr Ball mul(Ball a, Ball b) {
    const double mid = M::mul(a.mid, b.mid);
    if (unbounded(mid, a, b))
      return {mid, kInf};
    if (!is_finite(mid))
      return overflow(mid, least_of(a) * least_of(b));
    // A finite product has finite operands.
    const bool exact = a.mid == 0.0 || b.mid == 0.0;
    return {mid, abs(a.mid) * b.rad + abs(b.mid) * a.rad + a.rad * b.rad +
                     (exact ? 0.0 : abs(mid) * kHalfUlp + kTiny)};
  }
  [[gnu::always_inline]] static constexpr Ball div(Ball a, Ball b) {
    const double mid = M::div(a.mid, b.mid);
    if (unbounded(mid, a, b))
      return {mid, kInf};
    if (!is_finite(b.mid)) {
      // A finite value over an overflow: mid is ±0, and the exact quotient
      // is at most a's reach over b's least magnitude.
      const double least = least_of(b);
      if (!(least > 0.0))
        return {mid, kInf};
      return {mid, most_of(a) / least + kTiny};
    }
    if (b.rad >= abs(b.mid))
      return {mid, kInf};
    if (!is_finite(mid))
      return overflow(mid, least_of(a) / most_of(b));
    return {mid, (a.rad + abs(mid) * b.rad) / (abs(b.mid) - b.rad) +
                     abs(mid) * kHalfUlp + kTiny};
  }
  [[gnu::always_inline]] static constexpr Ball neg(Ball a) {
    return {-a.mid, a.rad};
  }
  // The operand Fn picks, but the other's error bounds the result too:
  // |max(A, B) - max(a, b)| <= max(|A - a|, |B - b|), and so for min.
  [[gnu::always_inline]] static constexpr Ball max(Ball a, Ball b) {
    const double mid = M::max(a.mid, b.mid);
    if (unbounded(mid, a, b))
      return {mid, kInf};
    if (is_finite(a.mid) && is_finite(b.mid))
      return {mid, max(a.rad, b.rad)};
    // With an overflow, by the operand Fn picks (p) and the other (q).
    const Ball p = (a.mid < b.mid) ? b : a, q = (a.mid < b.mid) ? a : b;
    if (p.mid == kInf) // max(P, Q) >= P
      return p;
    if (p.mid == -kInf) // both below -least
      return overflow(mid, min(least_of(p), least_of(q)));
    // q is below -least, which must not reach P
    return -least_of(q) <= p.mid - p.rad ? p : Ball{mid, kInf};
  }
  [[gnu::always_inline]] static constexpr Ball min(Ball a, Ball b) {
    // Picks as M::min does: b where b < a, else a.
    return neg(max(neg(a), neg(b)));
  }

  template <OpKind Op>
  [[gnu::always_inline]] static constexpr Ball unary(Ball a) {
    const double mid = M::template unary<Op>(a.mid);
    if (unbounded(mid, a))
      return {mid, kInf};
    if constexpr (Op == OpKind::Sin || Op == OpKind::Cos) {
      // |sin'| and |cos'| are at most 1 (and sin and cos of ±inf are NaN)
      return {mid, (a.rad < 2.0 ? a.rad : 2.0) + cx_error(mid)};
    } else {
      // exp, log, sqrt and erfc are monotone: the exact values over a's
      // range lie between those at its ends. Those are rounded outward: to
      // nearest, a.mid ± a.rad can land inside the range, or on a.mid itself
      // when a.rad is under half an ulp of it, and |f'| times what is lost
      // can be well over cx_error (exp(x) or erfc(x) at large x). An
      // overflow's range runs out to ±inf.
      if (a.rad == 0.0 && is_finite(a.mid) && is_finite(mid))
        return {mid, cx_error(mid)};
      double lo = a.mid == kInf    ? step_ulps(kMax - a.rad, -1)
                  : a.mid == -kInf ? -kInf
                                   : step_ulps(a.mid - a.rad, -1);
      const double hi = a.mid == -kInf  ? step_ulps(a.rad - kMax, 1)
                        : a.mid == kInf ? kInf
                                        : step_ulps(a.mid + a.rad, 1);
      if constexpr (Op == OpKind::Log) {
        if (!(lo > 0.0))
          return {mid, kInf};
      } else if constexpr (Op == OpKind::Sqrt) {
        lo = max(lo, 0.0);
      }
      const double f_lo = M::template unary<Op>(lo);
      const double f_hi = M::template unary<Op>(hi);
      if (is_finite(mid)) {
        if (!is_finite(f_lo) || !is_finite(f_hi))
          return {mid, kInf};
        return {mid, max(abs(f_hi - mid) + cx_error(f_hi),
                         abs(mid - f_lo) + cx_error(f_lo)) +
                         cx_error(mid)};
      }
      // An overflow, of exp, log or sqrt, which increase: at least f_lo,
      // less its error (cx_std overflows only past the largest double, give
      // or take that error).
      if (mid != kInf)
        return {mid, kInf};
      return overflow(mid, is_finite(f_lo) ? f_lo - cx_error(f_lo)
                                           : kMax - cx_error(kMax));
    }
  }
};

template <typename T>
using MathFor =
    std::conditional_t<std::is_same_v<T, Ball>, BallCxMath, IeeeCxMath>;

// Value of node I from the values of the nodes before it, with the target
// input at x: a double, or a Ball. Only plain copies of `n`'s fields appear
// in runtime code: naming the (consteval-only) Node there would make this
// function immediate.
template <info Fn, std::size_t I, std::size_t Target, typename T, std::size_t N,
          std::size_t NumArgs>
constexpr T node_value(const std::array<T, N> &val,
                       const std::array<double, NumArgs> &in, T x) {
  constexpr Node n = nodes_of<Fn>[I];
  constexpr OpKind op = n.op;
  constexpr std::size_t a = n.a, b = n.b, c = n.cond;
  if constexpr (op == OpKind::Input) {
    if constexpr (I == Target)
      return x;
    else
      return T{in[I]};
  } else if constexpr (op == OpKind::Const) {
    return T{static_cast<double>([:n.leaf:])};
  } else if constexpr (op_has_primal(op)) {
    return primal<op, MathFor<T>>(val[a], val[b], val[c]);
  } else {
    static_assert(false, "discontinuity_analysis: unsupported operation");
  }
}

// d(node I)/d(target). Every target-dependent node a crossing reads is affine
// (see Dependence), so only those ops need a rule, and at most one operand of
// a product varies.
template <info Fn, std::size_t I, std::size_t Target, std::size_t N>
constexpr double node_tangent(const std::array<double, N> &val,
                              const std::array<double, N> &tan) {
  using M = IeeeCxMath;
  constexpr Node n = nodes_of<Fn>[I];
  constexpr OpKind op = n.op;
  constexpr std::size_t a = n.a, b = n.b, c = n.cond;
  constexpr auto dep = target_dependence_of<Fn, Target>;
  if constexpr (!dep[I].varies)
    return 0.0;
  else if constexpr (op == OpKind::Input)
    return 1.0;
  else if constexpr (op == OpKind::Add)
    return M::add(tan[a], tan[b]);
  else if constexpr (op == OpKind::Sub)
    return M::sub(tan[a], tan[b]);
  else if constexpr (op == OpKind::Neg)
    return -tan[a];
  else if constexpr (op == OpKind::Output)
    return tan[a];
  else if constexpr (op == OpKind::Mul && dep[a].varies)
    return M::mul(tan[a], val[b]);
  else if constexpr (op == OpKind::Mul)
    return M::mul(val[a], tan[b]);
  else if constexpr (op == OpKind::Div)
    return M::div(tan[a], val[b]);
  else if constexpr (op == OpKind::Select && !dep[c].varies)
    return val[c] != 0.0 ? tan[a] : tan[b];
  else
    return 0.0; // not affine: analyze rejects any crossing that reads it
}

template <std::size_t N> struct Sweep {
  std::array<double, N> val = {};
  std::array<double, N> tan = {}; // derivative in the target
  // True (may_hold) or false (may_fail) for some target when read as a
  // condition. A node Fn reaches is one or both; one it never reaches is
  // neither.
  std::array<bool, N> may_hold = {}, may_fail = {};

  // Evaluated: Fn reaches node i for some target.
  constexpr bool reached(std::size_t i) const {
    return may_hold[i] || may_fail[i];
  }
};

// Values and target derivatives of every node needed to place the crossings,
// with the target input at x. A node is evaluated, as Fn evaluates it, only
// if its guard can hold for some target. Whether a condition may be true, and
// whether it may be false: a target-free one by its value, `!`, `&&` and `||`
// by their operands' (so `s > 1 && k > 0` at k = 0 never holds, and the else
// branch of `s > 1 || k > 0` at k = 1 is never taken), anything else that
// varies as either.
template <info Fn, std::size_t Target, std::size_t NumArgs>
constexpr auto sweep(const std::array<double, NumArgs> &in, double x) {
  static constexpr auto nodes = nodes_of<Fn>;
  static constexpr auto dep = target_dependence_of<Fn, Target>;
  static constexpr auto cone = crossing_cone_of<Fn, Target>;
  static constexpr auto needs_tangent = tangent_needed_of<Fn, Target>;
  Sweep<nodes.size()> s;
  template for (constexpr Node n : nodes) {
    if constexpr (cone[n.self]) {
      constexpr std::size_t i = n.self, guard = n.guard, a = n.a, b = n.b;
      constexpr OpKind op = n.op;
      if constexpr (guard != UNGUARDED) {
        if (!s.may_hold[guard])
          continue;
      }
      s.val[i] = node_value<Fn, i, Target>(s.val, in, x);
      if constexpr (needs_tangent[i])
        s.tan[i] = node_tangent<Fn, i, Target>(s.val, s.tan);
      if constexpr (!dep[i].varies) {
        s.may_hold[i] = s.val[i] != 0.0;
        s.may_fail[i] = !s.may_hold[i];
      } else if constexpr (op == OpKind::Not) {
        s.may_hold[i] = s.may_fail[a];
        s.may_fail[i] = s.may_hold[a];
      } else if constexpr (op == OpKind::And) {
        s.may_hold[i] = s.may_hold[a] && s.may_hold[b];
        s.may_fail[i] = s.may_fail[a] || s.may_fail[b];
      } else if constexpr (op == OpKind::Or) {
        s.may_hold[i] = s.may_hold[a] || s.may_hold[b];
        s.may_fail[i] = s.may_fail[a] && s.may_fail[b];
      } else {
        s.may_hold[i] = s.may_fail[i] = true;
      }
    }
  }
  return s;
}

// What crossing I tests the sign of, and its derivative in the target:
// lhs - rhs for a comparison, the value itself for a condition.
template <info Fn, std::size_t I, std::size_t N>
constexpr std::pair<double, double> gap(const Sweep<N> &s) {
  constexpr Node n = nodes_of<Fn>[I];
  constexpr std::size_t a = n.a, b = n.b;
  if constexpr (is_comparison(n.op))
    return {IeeeCxMath::sub(s.val[a], s.val[b]),
            IeeeCxMath::sub(s.tan[a], s.tan[b])};
  else
    return {s.val[I], s.tan[I]};
}

// The nodes crossing I's gap reads, in order.
template <info Fn, std::size_t I>
consteval std::vector<std::size_t> gap_cone() {
  const std::vector<char> read = gap_reads<Fn>(I);
  std::vector<std::size_t> cone;
  for (std::size_t i = 0; i <= I; ++i)
    if (read[i])
      cone.push_back(i);
  return cone;
}

template <info Fn, std::size_t I>
inline constexpr auto gap_cone_of = std::define_static_array(gap_cone<Fn, I>());

// Whether crossing I's gap, in a sweep where it is not finite, is so because
// arithmetic on the target overflowed there: a +, -, * or / that depends on
// the target, or the comparison's own lhs - rhs, of finite values. Otherwise
// a target-free value is not finite, and so is the gap for every target.
template <info Fn, std::size_t Target, std::size_t I, std::size_t N>
constexpr bool target_overflows(const Sweep<N> &s) {
  static constexpr auto nodes = nodes_of<Fn>;
  static constexpr auto dep = target_dependence_of<Fn, Target>;
  constexpr OpKind crossing_op = nodes[I].op;
  constexpr std::size_t lhs = nodes[I].a, rhs = nodes[I].b;
  if constexpr (is_comparison(crossing_op)) {
    if (is_finite(s.val[lhs]) && is_finite(s.val[rhs]))
      return true;
  }
  bool overflowed = false;
  template for (constexpr std::size_t j : gap_cone_of<Fn, I>) {
    constexpr OpKind op = nodes[j].op;
    constexpr std::size_t a = nodes[j].a, b = nodes[j].b;
    if constexpr (dep[j].varies && (op == OpKind::Add || op == OpKind::Sub ||
                                    op == OpKind::Mul || op == OpKind::Div)) {
      if (!is_finite(s.val[j]) && is_finite(s.val[a]) && is_finite(s.val[b]))
        overflowed = true;
    }
  }
  return overflowed;
}

// Crossing I's gap (as gap() reads it from a sweep) with the target at x, a
// double or a Ball, evaluating only the nodes it reads. A node is skipped, as
// sweep() skips it, if its guard can never hold; `reach` is any sweep, since
// that does not depend on x.
template <info Fn, std::size_t Target, std::size_t I, typename T, std::size_t N,
          std::size_t NumArgs>
constexpr T gap_at(const std::array<double, NumArgs> &in, T x,
                   const Sweep<N> &reach) {
  static constexpr auto nodes = nodes_of<Fn>;
  std::array<T, N> val = {};
  template for (constexpr std::size_t j : gap_cone_of<Fn, I>) {
    constexpr std::size_t guard = nodes[j].guard;
    if constexpr (guard != UNGUARDED) {
      if (!reach.may_hold[guard])
        continue;
    }
    val[j] = node_value<Fn, j, Target>(val, in, x);
  }
  constexpr std::size_t a = nodes[I].a, b = nodes[I].b;
  if constexpr (is_comparison(nodes[I].op))
    return MathFor<T>::sub(val[a], val[b]);
  else
    return val[I];
}

// How far a root is moved to where crossing I's sides are equal (see snap).
inline constexpr int kSnapUlps = 4;
// How near the next root must be to be part of the same point (see analyze).
inline constexpr int kClusterUlps = 4;

// Trailing zero bits in x's significand: more means a shorter number.
constexpr int roundness(double x) {
  const std::uint64_t significand =
      std::bit_cast<std::uint64_t>(x) & ((std::uint64_t{1} << 52) - 1);
  return significand == 0 ? 52 : std::countr_zero(significand);
}

// -g(0)/g' rounds, and Fn's own arithmetic can make the two sides equal on a
// short run of doubles rather than one: `s / 3 > k / 3` at k = 100 solves to
// 100.00000000000001, and s / 3 == k / 3 at both that and 100. Of the
// doubles within kSnapUlps where g is exactly 0, take the shortest (the
// nearest on a tie, then the one above): that is the exact root when it is a
// double, and the same choice for every crossing that has it, so crossings
// that coincide in exact arithmetic coincide here too and are measured
// together. With no such double, r stands. The candidates are tried in that
// order of preference, so the search stops at the first zero. Each candidate
// evaluates only what crossing I reads (gap_at), not every crossing.
template <info Fn, std::size_t Target, std::size_t I, std::size_t N,
          std::size_t NumArgs>
constexpr double snap(const std::array<double, NumArgs> &in, double r,
                      const Sweep<N> &reach) {
  std::array<double, 2 * kSnapUlps + 1> candidates = {};
  std::size_t count = 0;
  for (int k = 0; k <= kSnapUlps; ++k) {
    for (int dir = 1; dir >= -1; dir -= 2) {
      if (k == 0 && dir < 0)
        continue;
      // Insert after every candidate at least as short: a stable sort,
      // shortest first, keeping nearest-then-above among equals.
      const double x = step_ulps(r, dir * k);
      std::size_t at = count++;
      for (; at > 0 && roundness(candidates[at - 1]) < roundness(x); --at)
        candidates[at] = candidates[at - 1];
      candidates[at] = x;
    }
  }
  for (const double x : candidates)
    if (is_finite(x) && gap_at<Fn, Target, I>(in, x, reach) == 0.0)
      return x;
  return r;
}

// Fn's value with the target input at x, taking branches as Fn does (a
// guarded node is skipped when its guard is false), except where forced[i]
// is 0 or 1: comparison i takes that outcome, and a condition i reads as
// that truth wherever it is used as one.
template <info Fn, std::size_t Target, typename T, std::size_t N,
          std::size_t NumArgs>
constexpr T value_with(const std::array<double, NumArgs> &in, T x,
                       const std::array<signed char, N> &forced) {
  using Math = MathFor<T>;
  static constexpr auto nodes = nodes_of<Fn>;
  std::array<T, N> val = {};
  // Node j read as a condition: its forced truth, else its value (nonzero is
  // true).
  const auto cond = [&](std::size_t j) {
    return forced[j] >= 0 ? T{static_cast<double>(forced[j])} : val[j];
  };
  template for (constexpr Node n : nodes) {
    constexpr std::size_t i = n.self, guard = n.guard, a = n.a, b = n.b,
                          c = n.cond;
    constexpr OpKind op = n.op;
    if constexpr (guard != UNGUARDED) {
      if (mid_of(cond(guard)) == 0.0)
        continue;
    }
    if constexpr (is_comparison(op)) {
      if (forced[i] >= 0) {
        val[i] = T{static_cast<double>(forced[i])};
        continue;
      }
    }
    if constexpr (op == OpKind::Not || op == OpKind::And || op == OpKind::Or)
      val[i] = primal<op, Math>(cond(a), cond(b), T{});
    else if constexpr (op == OpKind::Select)
      val[i] = primal<op, Math>(val[a], val[b], cond(c));
    else
      val[i] = node_value<Fn, i, Target>(val, in, x);
  }
  return val[N - 1];
}

// The analysis behind every entry point.
//
// Each crossing compares g with 0 (g = lhs - rhs, or the condition's value),
// and g is affine in the target, so its value and slope at 0 place the root
// exactly and say which outcome holds just above it. Roots of ordering
// comparisons a few ulps apart or less are one point. At each point, the
// jump is Fn's right limit minus its left limit: Fn evaluated at the point
// with every crossing rooted there forced to its outcome just above, minus
// the same with the outcome just below. Whatever sits between the comparison
// and the output (`!`, `&&`, `||`, nested selects, scaling, other jumps at
// the same point) is evaluated rather than pattern-matched. `==` / `!=` and
// conditions rooted there take their outcome off the point on both sides.
// Points with no jump are not reported, and nor are those whose jump is
// within the rounding error of evaluating Fn there (as at a kink whose root is
// not a double): Fn is evaluated in Balls, over every target its crossings'
// exact roots could be at, and the jump must exceed its error bound.
//
// `in` holds the function's arguments; the target's slot is ignored.
template <info Fn, std::size_t Target, std::size_t MaxPoints,
          std::size_t NumArgs>
constexpr DiscontinuityPointsWithAmplitudes<MaxPoints>
analyze(const std::array<double, NumArgs> &in) {
  static constexpr auto nodes = nodes_of<Fn>;
  static constexpr auto dep = target_dependence_of<Fn, Target>;
  static constexpr auto crossing = crossings_of<Fn, Target>;
  static_assert(Target < input_count_of<Fn>(),
                "discontinuity_analysis: the target index is not an argument "
                "of the function");
  static_assert(input_count_of<Fn>() == NumArgs,
                "discontinuity_analysis: pass one fixed value for each "
                "argument except the target, in order");
  constexpr std::size_t N = nodes.size();

  // Per crossing slot: whether it flips at all, where, its outcome just
  // below and just above that point, and whether it can make a jump there.
  std::array<bool, N> rooted = {}, makes_point = {};
  std::array<double, N> root = {}, slope_of = {};
  std::array<signed char, N> below = {}, above = {};

  const auto at_zero = sweep<Fn, Target>(in, 0.0);
  template for (constexpr Node n : nodes) {
    if constexpr (crossing[n.self]) {
      constexpr std::size_t i = n.self;
      constexpr OpKind op = n.op;
      if constexpr (is_comparison(op))
        static_assert(dep[n.a].affine && dep[n.b].affine,
                      "discontinuity_analysis: both sides of a comparison on "
                      "the target must be affine in it (built from it with + "
                      "- and * / by values that do not depend on it, and ?: "
                      "on conditions that do not)");
      else
        static_assert(dep[i].affine,
                      "discontinuity_analysis: a value used as a condition "
                      "that varies continuously with the target must be "
                      "affine in it (built from it with + - and * / by "
                      "values that do not depend on it, and ?: on conditions "
                      "that do not)");
      const auto [g0, slope] = gap<Fn, i>(at_zero);
      // A slope that is not finite means a coefficient inside g is infinite
      // (`s / k > 1` at k = 0): g is then ±inf or NaN except where that
      // term's own value crosses 0, which is not solved for. An error rather
      // than a missed jump, even where g turns out never to flip (`s * (k -
      // k) > 0` at k = inf). Once a coefficient is infinite, no affine op
      // makes it finite again, so with a finite slope every one is finite.
      if (at_zero.reached(i) && !is_finite(slope))
        throw "discontinuity_analysis: a crossing's slope in the target is "
              "not finite (as for s / k > 1 at k = 0), so its point cannot "
              "be placed";
      // A constant that is not finite because Fn's arithmetic on the target
      // overflowed at 0 (`(s - k) * 1e10 > 0` at k = 1e300, which flips at
      // s = k) is not the one that places the root, which then cannot be
      // placed: an error, not a missed jump.
      if (at_zero.reached(i) && !is_finite(g0) &&
          target_overflows<Fn, Target, i>(at_zero))
        throw "discontinuity_analysis: a crossing's sides overflow with the "
              "target at 0 (as (s - k) * 1e10 > 0 does at k = 1e300), so its "
              "point cannot be placed";
      // No root if Fn never reaches the crossing, if g is flat (`s * k > 1`
      // at k = 0) or, with every coefficient finite, its constant is not
      // because a target-free value is not (`s > 1 / k` at k = 0: -inf for
      // every target, never true), or if the root is beyond the doubles.
      const double r =
          at_zero.reached(i) && slope != 0.0 && is_finite(g0)
              ? -(g0 / slope) + 0.0 // `+ 0.0` turns a -0 root into 0
              : kNaN;
      if (is_finite(r)) {
        rooted[i] = true;
        makes_point[i] = is_ordering(op);
        root[i] = snap<Fn, Target, i>(in, r, at_zero);
        slope_of[i] = slope;
        // Just above the root, g has the sign of its slope.
        const bool g_positive = slope > 0.0;
        if constexpr (op == OpKind::Gt || op == OpKind::Ge) {
          above[i] = g_positive;
          below[i] = !g_positive;
        } else if constexpr (op == OpKind::Lt || op == OpKind::Le) {
          above[i] = !g_positive;
          below[i] = g_positive;
        } else {
          // == is false off its point; != and a condition (v != 0) are true
          above[i] = below[i] = (op != OpKind::Eq);
        }
      }
    }
  }

  // The crossings that can make a point, in order of their roots.
  std::array<std::size_t, N> order = {};
  std::size_t ordered = 0;
  for (std::size_t c = 0; c < N; ++c) {
    if (!makes_point[c])
      continue;
    std::size_t at = ordered++;
    for (; at > 0 && root[c] < root[order[at - 1]]; --at)
      order[at] = order[at - 1];
    order[at] = c;
  }

  // Roots each within kClusterUlps of the one before are one point, measured
  // once: they are often one crossing in exact arithmetic (`s * 1.1 > k`
  // and `s / 0.9 > k / 0.99`), and Fn's own arithmetic can make a
  // comparison's sides exactly equal on a run of doubles between them, so
  // measured apart each could lose the other's share of the jump. The point
  // is the shortest of their roots (as in snap), the lowest on a tie. Group g
  // is order[first[g]..last[g]].
  std::array<std::size_t, N> group_of = {}, first = {}, last = {};
  std::array<double, N> point = {};
  std::size_t groups = 0;
  for (std::size_t k = 0; k < ordered; ++groups) {
    first[groups] = k;
    point[groups] = root[order[k]];
    group_of[order[k]] = groups;
    while (k + 1 < ordered &&
           root[order[k + 1]] <= step_ulps(root[order[k]], kClusterUlps)) {
      ++k;
      group_of[order[k]] = groups;
      if (roundness(root[order[k]]) > roundness(point[groups]))
        point[groups] = root[order[k]];
    }
    last[groups] = k++;
  }

  // How far from its point each group's exact roots can be: each one's gap
  // is affine, so its root is its gap at the point, give or take the gap's
  // rounding error, over its slope (0 where that gap is computed exactly).
  // One pass over the crossings, each adding to its own group's.
  std::array<double, N> roots_within = {};
  template for (constexpr std::size_t i : ordering_crossings_of<Fn, Target>) {
    if (makes_point[i]) {
      const std::size_t g = group_of[i];
      const Ball gap = gap_at<Fn, Target, i>(in, Ball{point[g], 0.0}, at_zero);
      const double d = IeeeCxMath::add(BallCxMath::abs(gap.mid), gap.rad);
      const double within =
          BallCxMath::abs(slope_of[i]) > 0.0
              ? IeeeCxMath::div(d, BallCxMath::abs(slope_of[i]))
              : kInf;
      roots_within[g] =
          within == within ? BallCxMath::max(roots_within[g], within) : kInf;
    }
  }

  DiscontinuityPointsWithAmplitudes<MaxPoints> result;
  for (std::size_t g = 0; g < groups; ++g) {
    // Every crossing rooted in the group's span takes its outcome above it
    // on the right and below it on the left: `==` / `!=` and conditions,
    // their outcome off the point on both.
    const double from = step_ulps(root[order[first[g]]], -kClusterUlps),
                 to = step_ulps(root[order[last[g]]], kClusterUlps);
    std::array<signed char, N> right, left;
    right.fill(-1);
    left.fill(-1);
    for (std::size_t j = 0; j < N; ++j) {
      if (!rooted[j] || root[j] < from || to < root[j])
        continue;
      right[j] = above[j];
      left[j] = below[j];
    }
    // Fn's right and left limits there, and how far from them the exact
    // values can be with the target anywhere the roots can be.
    const Ball target{point[g], roots_within[g]};
    const Ball from_right = value_with<Fn, Target>(in, target, right);
    const Ball from_left = value_with<Fn, Target>(in, target, left);
    if (!is_finite(from_right.mid) || !is_finite(from_left.mid))
      throw "discontinuity_analysis: Fn is not finite on one side of a "
            "crossing point (a pole or an undefined branch there, not a "
            "jump)";
    const double jump = from_right.mid - from_left.mid;
    // A kink -- continuous where its branches meet -- has no jump at its
    // exact root, but that need not be a double, nor the point (snap finds a
    // double where the crossing's sides are equal, not the branches'), so
    // rounding can leave a tiny one at the point, of either sign and any
    // order (`s > k / 3 ? (3 * s - k) * (3 * s - k) : 0`). A jump is reported
    // only where it exceeds the two sides' error bounds, which cover that,
    // with a factor of 2 for the bounds' own rounding and cx_std's measured
    // (not proven) accuracy. Where the roots and branches are computed
    // exactly the bound is 0, and any jump is reported (`s > k ? s - k + 1 :
    // 0` at k = 1e16).
    const double bound = 2.0 * (from_right.rad + from_left.rad);
    // An unbounded side -- a value on the way with no bound on its error, as
    // a divisor that may be 0 (`1 / (k * 0.1 * 10 - k)` at k = 3, which is
    // rounding alone) or a NaN, though Fn is finite -- cannot tell them
    // apart: an error, not a silently dropped jump. (An overflow alone is
    // bounded: `1 / (1 + exp(k))` at k = 1000 is within 1e-308 of 0.)
    if (jump != 0.0 && bound == kInf)
      throw "discontinuity_analysis: Fn's error near a crossing point cannot "
            "be bounded (a value on the way there has none, as for a divisor "
            "that may be 0), so its jump cannot be told from rounding";
    if (jump != 0.0 && BallCxMath::abs(jump) > bound)
      result.add_point_with_amplitude(point[g], jump);
  }

  return result;
}

} // namespace detail_disc

template <info Fn, std::size_t TargetArgIndex, std::size_t MaxPoints = 16,
          typename... FixedArgs>
consteval DiscontinuityPoints<MaxPoints>
get_discontinuity_points(FixedArgs... fixed_args) {
  const auto found = detail_disc::analyze<Fn, TargetArgIndex, MaxPoints>(
      detail_disc::make_inputs<TargetArgIndex>(fixed_args...));
  DiscontinuityPoints<MaxPoints> result;
  for (std::size_t i = 0; i < found.size(); ++i)
    result.points[i] = found.point(i);
  result.count = found.size();
  return result;
}

template <info Fn, std::size_t TargetArgIndex, std::size_t MaxPoints = 16,
          typename... FixedArgs>
consteval DiscontinuityPointsWithAmplitudes<MaxPoints>
get_discontinuity_points_and_amplitudes(FixedArgs... fixed_args) {
  return detail_disc::analyze<Fn, TargetArgIndex, MaxPoints>(
      detail_disc::make_inputs<TargetArgIndex>(fixed_args...));
}

// Runtime version: accepts dynamic (non-constexpr) fixed arguments. The DAG is
// still analysed at compile time; only the arithmetic on the argument values
// runs at runtime.
template <info Fn, std::size_t TargetArgIndex, std::size_t MaxPoints = 16,
          typename... FixedArgs>
inline DiscontinuityPointsWithAmplitudes<MaxPoints>
get_discontinuity_points_and_amplitudes_rt(FixedArgs... fixed_args) {
  return detail_disc::analyze<Fn, TargetArgIndex, MaxPoints>(
      detail_disc::make_inputs<TargetArgIndex>(fixed_args...));
}

} // namespace ad

#endif // DISCONTINUITY_ANALYSIS_HPP
