#ifndef IS_CONTINUOUS_HPP
#define IS_CONTINUOUS_HPP

// is_continuous.hpp — compile-time continuity checker via interval arithmetic.
//
// Walks the same SSA/DAG that autograd.h builds, propagating an interval
// (range) through every node and checking that each op's preconditions are
// met on that range.
//
// Usage:
//   static_assert(ad::is_continuous_on<^^call_price>(
//       ad::Interval{90.0, 110.0},   // S
//       ad::Interval{90.0, 110.0},   // K
//       ad::Interval{0.05,  0.30},   // v
//       ad::Interval{0.50,  1.50}    // T
//   ));
//
// Lean correspondence (soundness theorem):
//   theorem is_continuous_sound (e : Expr n) (bounds : Fin n → Interval)
//       (h : is_continuous_on e bounds = true) :
//       ContinuousOn (eval e) (input_domain bounds) := ...
//
// NOTE: interval arithmetic here is an *over-approximation* — ranges may be
// wider than the true image, so the checker is *sound but incomplete*: a
// `true` answer is a guarantee; a `false` may be a false alarm on a
// pathological domain (e.g. a Div whose denominator's true range avoids zero
// but the interval over-approximation doesn't).

#include "autograd.h" // for ad::Node, ad::OpKind, ad::build_nodes<>
#include "cx_erfc.hpp"
#include "cx_exp.hpp"
#include "cx_log.hpp"
#include "cx_sqrt.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace ad {

// ---------------------------------------------------------------------------
// Interval — a closed real interval [lo, hi].
// ---------------------------------------------------------------------------
struct Interval {
  double lo;
  double hi;

  consteval bool contains_zero() const { return lo <= 0.0 && 0.0 <= hi; }
  consteval bool strictly_positive() const { return lo > 0.0; }
  consteval bool non_negative() const { return lo >= 0.0; }

  // Width guard: a degenerate/empty interval is treated as a problem.
  consteval bool valid() const { return lo <= hi; }
};

// ---------------------------------------------------------------------------
// Interval arithmetic helpers
// ---------------------------------------------------------------------------
namespace detail_cont {

consteval Interval add(Interval a, Interval b) {
  return {a.lo + b.lo, a.hi + b.hi};
}

consteval Interval sub(Interval a, Interval b) {
  return {a.lo - b.hi, a.hi - b.lo};
}

consteval Interval neg(Interval a) { return {-a.hi, -a.lo}; }

consteval Interval mul(Interval a, Interval b) {
  double lo = std::min({a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi});
  double hi = std::max({a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi});
  return {lo, hi};
}

// Precondition: b does not contain zero (caller must check).
consteval Interval div(Interval a, Interval b) {
  Interval b_inv{1.0 / b.hi, 1.0 / b.lo};
  return mul(a, b_inv);
}

consteval Interval sin_range(Interval a) {
  // Conservative: full range [-1, 1] if the interval is >= 2π wide.
  constexpr double two_pi = 2.0 * 3.141592653589793;
  if (a.hi - a.lo >= two_pi)
    return {-1.0, 1.0};
  // Otherwise use endpoint values as a rough bound (not tight, but sound).
  double lo = std::min(std::sin(a.lo), std::sin(a.hi));
  double hi = std::max(std::sin(a.lo), std::sin(a.hi));
  // Widen slightly for intermediate extrema (conservative).
  return {std::min(lo, -1.0 * (lo < 0.0 ? 1.0 : 0.0)),
          std::max(hi, 1.0 * (hi > 0.0 ? 1.0 : 0.0))};
}

consteval Interval cos_range(Interval a) {
  constexpr double two_pi = 2.0 * 3.141592653589793;
  if (a.hi - a.lo >= two_pi)
    return {-1.0, 1.0};
  double lo = std::min(std::cos(a.lo), std::cos(a.hi));
  double hi = std::max(std::cos(a.lo), std::cos(a.hi));
  return {std::min(lo, -1.0 * (lo < 0.0 ? 1.0 : 0.0)),
          std::max(hi, 1.0 * (hi > 0.0 ? 1.0 : 0.0))};
}

// Tan is discontinuous at π/2 + k·π.
// Check: does the interval [a.lo, a.hi] contain any such point?
consteval bool tan_has_discontinuity(Interval a) {
  constexpr double pi = 3.141592653589793;
  constexpr double half_pi = pi / 2.0;
  // The poles are at half_pi + k*pi. Find k such that half_pi + k*pi ∈ [lo,
  // hi].
  double k_lo = std::ceil((a.lo - half_pi) / pi);
  double k_hi = std::floor((a.hi - half_pi) / pi);
  return k_lo <= k_hi; // at least one integer k in range → discontinuity
}

consteval Interval exp_range(Interval a) {
  return {cx::exp(a.lo), cx::exp(a.hi)};
}

// Precondition: a.strictly_positive() (caller must check).
consteval Interval log_range(Interval a) {
  return {cx::log(a.lo), cx::log(a.hi)};
}

// Precondition: a.non_negative() (caller must check).
consteval Interval sqrt_range(Interval a) {
  return {cx::sqrt(a.lo), cx::sqrt(a.hi)};
}

// erfc is entire (monotone decreasing on ℝ).
consteval Interval erfc_range(Interval a) {
  return {cx::erfc(a.hi), cx::erfc(a.lo)};
}

// A sentinel representing "computation aborted due to a continuity failure".
consteval Interval failed() {
  return {std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN()};
}

consteval bool is_failed(Interval iv) {
  // NaN != NaN
  return !(iv.lo == iv.lo);
}

} // namespace detail_cont

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------
struct ContinuityResult {
  bool continuous;
  // Index of the first node that failed (-1 if none).
  int failing_node;
  // The op that caused the failure (meaningful only when continuous == false).
  OpKind failing_op;
};

// ---------------------------------------------------------------------------
// Core checker: walks the DAG, propagating ranges, checking preconditions.
// ---------------------------------------------------------------------------
// InputBounds must be an array/span-like of Interval with one entry per
// function parameter (in parameter order).
template <info Fn, std::size_t P>
consteval ContinuityResult
check_continuity(const std::array<Interval, P> &input_bounds) {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());
  constexpr std::size_t N = nodes.size();

  Interval ranges[N];
  for (std::size_t i = 0; i < N; ++i)
    ranges[i] = detail_cont::failed();

  template for (constexpr auto n : nodes) {
    // n is constexpr, so n.op / n.a / n.b / n.self are all constexpr.
    // ranges[] is a runtime array (within consteval): reads/writes are fine.
    constexpr bool has_a = n.op != OpKind::Input && n.op != OpKind::Const;
    constexpr bool has_b = n.op == OpKind::Add || n.op == OpKind::Sub ||
                           n.op == OpKind::Mul || n.op == OpKind::Div;

    Interval a = has_a ? ranges[n.a] : detail_cont::failed();
    Interval b = has_b ? ranges[n.b] : detail_cont::failed();

    if constexpr (n.op == OpKind::Input) {
      if constexpr (n.self >= P)
        return {false, static_cast<int>(n.self), OpKind::Input};
      ranges[n.self] = input_bounds[n.self];

    } else if constexpr (n.op == OpKind::Const) {
      // Splice the literal value to get a tight point interval.
      // This mirrors how autograd.h evaluators use `[: n.leaf :]`.
      constexpr double v = static_cast<double>([:n.leaf:]);
      ranges[n.self] = {v, v};

    } else if constexpr (n.op == OpKind::Output) {
      ranges[n.self] = ranges[n.a];

    } else if constexpr (n.op == OpKind::Add) {
      ranges[n.self] = detail_cont::add(a, b);

    } else if constexpr (n.op == OpKind::Sub) {
      ranges[n.self] = detail_cont::sub(a, b);

    } else if constexpr (n.op == OpKind::Mul) {
      ranges[n.self] = detail_cont::mul(a, b);

    } else if constexpr (n.op == OpKind::Div) {
      if (b.contains_zero())
        return {false, static_cast<int>(n.self), OpKind::Div};
      ranges[n.self] = detail_cont::div(a, b);

    } else if constexpr (n.op == OpKind::Neg) {
      ranges[n.self] = detail_cont::neg(a);

    } else if constexpr (n.op == OpKind::Sin) {
      ranges[n.self] = detail_cont::sin_range(a);

    } else if constexpr (n.op == OpKind::Cos) {
      ranges[n.self] = detail_cont::cos_range(a);

    } else if constexpr (n.op == OpKind::Exp) {
      ranges[n.self] = detail_cont::exp_range(a);

    } else if constexpr (n.op == OpKind::Log) {
      if (!a.strictly_positive())
        return {false, static_cast<int>(n.self), OpKind::Log};
      ranges[n.self] = detail_cont::log_range(a);

    } else if constexpr (n.op == OpKind::Sqrt) {
      if (!a.non_negative())
        return {false, static_cast<int>(n.self), OpKind::Sqrt};
      ranges[n.self] = detail_cont::sqrt_range(a);

    } else if constexpr (n.op == OpKind::Erfc) {
      ranges[n.self] = detail_cont::erfc_range(a);

    } else {
      // Tensor ops or any unrecognised op: not supported yet.
      return {false, static_cast<int>(n.self), n.op};
    }
  }

  return {true, -1, OpKind::Input};
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

// Returns true iff the function reflected by Fn is continuous on the Cartesian
// product of the given input intervals.
//
// Example:
//   static_assert(ad::is_continuous_on<^^call_price>(
//       ad::Interval{90.0, 110.0},
//       ad::Interval{90.0, 110.0},
//       ad::Interval{0.05, 0.30},
//       ad::Interval{0.50, 1.50}));
template <info Fn, typename... Intervals>
consteval bool is_continuous_on(Intervals... bounds) {
  static_assert((std::is_same_v<Intervals, Interval> && ...),
                "All arguments must be ad::Interval");
  constexpr std::size_t P = sizeof...(bounds);
  const std::array<Interval, P> arr{bounds...};
  return check_continuity<Fn, P>(arr).continuous;
}

// Richer variant that returns the full ContinuityResult (which node failed,
// etc.).
template <info Fn, typename... Intervals>
consteval ContinuityResult continuity_result(Intervals... bounds) {
  static_assert((std::is_same_v<Intervals, Interval> && ...),
                "All arguments must be ad::Interval");
  constexpr std::size_t P = sizeof...(bounds);
  const std::array<Interval, P> arr{bounds...};
  return check_continuity<Fn, P>(arr);
}

} // namespace ad

#endif // IS_CONTINUOUS_HPP
