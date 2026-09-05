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
//
// Branches are decided over the box: a provably true/false `c` checks only the
// live branch (`x > 0 ? log(x) : 0.0` is continuous on [-2,-1]); an undecidable
// one may jump, and is reported. abs/max/min are ops, not branches: kinked but
// continuous. Not handled: Sin/Cos narrower than 2π — see sin_range.

#include "../autograd.h" // for ad::Node, ad::OpKind, ad::build_nodes<>
#include "../cx_std/cx_erfc.hpp"
#include "../cx_std/cx_exp.hpp"
#include "../cx_std/cx_log.hpp"
#include "../cx_std/cx_sqrt.hpp"

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

// Truth of a condition over the box; `Unknown` = it straddles the switch.
enum class Tri { False, True, Unknown };

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

// LIMITATION: only the `>= 2π` early return works -- the narrow path calls
// non-constexpr std::sin/std::cos, so it fails to compile rather than answer
// wrongly. `trig` passes only because it takes the early return. Fix: add
// cx_std/cx_sin.hpp + cx_cos.hpp, and tighten these loose bounds.
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

// The kinks: continuous everywhere, so none has a precondition.
consteval Interval abs_range(Interval a) {
  if (a.lo >= 0.0)
    return a;
  if (a.hi <= 0.0)
    return {-a.hi, -a.lo};
  return {0.0, std::max(-a.lo, a.hi)};
}

consteval Interval max_range(Interval a, Interval b) {
  return {std::max(a.lo, b.lo), std::max(a.hi, b.hi)};
}

consteval Interval min_range(Interval a, Interval b) {
  return {std::min(a.lo, b.lo), std::min(a.hi, b.hi)};
}

// --- Kleene three-valued logic, and comparisons lifted to intervals ---------
consteval Tri tri_not(Tri t) {
  if (t == Tri::True)
    return Tri::False;
  if (t == Tri::False)
    return Tri::True;
  return Tri::Unknown;
}

consteval Tri tri_and(Tri x, Tri y) {
  if (x == Tri::False || y == Tri::False)
    return Tri::False;
  if (x == Tri::True && y == Tri::True)
    return Tri::True;
  return Tri::Unknown;
}

consteval Tri tri_or(Tri x, Tri y) {
  if (x == Tri::True || y == Tri::True)
    return Tri::True;
  if (x == Tri::False && y == Tri::False)
    return Tri::False;
  return Tri::Unknown;
}

consteval Tri tri_lt(Interval a, Interval b) {
  if (a.hi < b.lo)
    return Tri::True;
  if (a.lo >= b.hi)
    return Tri::False;
  return Tri::Unknown;
}

consteval Tri tri_le(Interval a, Interval b) {
  if (a.hi <= b.lo)
    return Tri::True;
  if (a.lo > b.hi)
    return Tri::False;
  return Tri::Unknown;
}

// Decidable only between point or disjoint intervals.
consteval Tri tri_eq(Interval a, Interval b) {
  if (a.lo == a.hi && b.lo == b.hi && a.lo == b.lo)
    return Tri::True;
  if (a.hi < b.lo || b.hi < a.lo)
    return Tri::False;
  return Tri::Unknown;
}

// Placeholder for nodes with no meaningful range; `poisoned[]` tracks why.
consteval Interval failed() {
  return {std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN()};
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
  Tri truth[N];
  // A precondition failed below this node under an undecidable guard. Explicit
  // rather than a NaN range: `std::min(2.0, NaN)` is `2.0`, laundering it away.
  bool poisoned[N];
  for (std::size_t i = 0; i < N; ++i) {
    ranges[i] = detail_cont::failed();
    truth[i] = Tri::Unknown;
    poisoned[i] = false;
  }

  template for (constexpr auto n : nodes) {
    // n is constexpr, so n.op / n.a / n.b / n.self are all constexpr.
    // ranges[] is a runtime array (within consteval): reads/writes are fine.

    // Guard first: branch nodes precede their Select, so otherwise a
    // provably-untaken branch is checked as if it ran.
    Tri guard = Tri::True;
    if constexpr (n.guard != UNGUARDED)
      guard = truth[n.guard];

    if (guard == Tri::False) {
      // Unreachable. False (not Unknown) propagates deadness into nested
      // guards.
      ranges[n.self] = detail_cont::failed();
      truth[n.self] = Tri::False;
      poisoned[n.self] = false; // never read: nothing live consumes a dead node

    } else {
      // Only a node we can prove runs has its preconditions enforced; under an
      // undecidable guard, poison it instead and let the Select report.
      const bool certainly_runs = (guard == Tri::True);

      // Inherited; the decision points (comparisons, Select, Output) refuse it.
      const bool operand_poisoned = (op_has_a(n.op) && poisoned[n.a]) ||
                                    (op_has_b(n.op) && poisoned[n.b]) ||
                                    (op_has_cond(n.op) && poisoned[n.cond]);
      poisoned[n.self] = operand_poisoned;

      Interval a = op_has_a(n.op) ? ranges[n.a] : detail_cont::failed();
      Interval b = op_has_b(n.op) ? ranges[n.b] : detail_cont::failed();

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
        if (poisoned[n.a])
          return {false, static_cast<int>(n.self), OpKind::Output};
        ranges[n.self] = ranges[n.a];

      } else if constexpr (n.op == OpKind::Add) {
        ranges[n.self] = detail_cont::add(a, b);

      } else if constexpr (n.op == OpKind::Sub) {
        ranges[n.self] = detail_cont::sub(a, b);

      } else if constexpr (n.op == OpKind::Mul) {
        ranges[n.self] = detail_cont::mul(a, b);

      } else if constexpr (n.op == OpKind::Div) {
        if (b.contains_zero()) {
          if (certainly_runs)
            return {false, static_cast<int>(n.self), OpKind::Div};
          ranges[n.self] = detail_cont::failed();
          poisoned[n.self] = true;
        } else {
          ranges[n.self] = detail_cont::div(a, b);
        }

      } else if constexpr (n.op == OpKind::Neg) {
        ranges[n.self] = detail_cont::neg(a);

      } else if constexpr (n.op == OpKind::Sin) {
        ranges[n.self] = detail_cont::sin_range(a);

      } else if constexpr (n.op == OpKind::Cos) {
        ranges[n.self] = detail_cont::cos_range(a);

      } else if constexpr (n.op == OpKind::Exp) {
        ranges[n.self] = detail_cont::exp_range(a);

      } else if constexpr (n.op == OpKind::Log) {
        if (!a.strictly_positive()) {
          if (certainly_runs)
            return {false, static_cast<int>(n.self), OpKind::Log};
          ranges[n.self] = detail_cont::failed();
          poisoned[n.self] = true;
        } else {
          ranges[n.self] = detail_cont::log_range(a);
        }

      } else if constexpr (n.op == OpKind::Sqrt) {
        if (!a.non_negative()) {
          if (certainly_runs)
            return {false, static_cast<int>(n.self), OpKind::Sqrt};
          ranges[n.self] = detail_cont::failed();
          poisoned[n.self] = true;
        } else {
          ranges[n.self] = detail_cont::sqrt_range(a);
        }

      } else if constexpr (n.op == OpKind::Erfc) {
        ranges[n.self] = detail_cont::erfc_range(a);

        // --- the kinks: continuous everywhere, no precondition ---------------
      } else if constexpr (n.op == OpKind::Abs) {
        ranges[n.self] = detail_cont::abs_range(a);

      } else if constexpr (n.op == OpKind::Max) {
        ranges[n.self] = detail_cont::max_range(a, b);

      } else if constexpr (n.op == OpKind::Min) {
        ranges[n.self] = detail_cont::min_range(a, b);

        // --- conditions: carry a truth value rather than a useful range ------
      } else if constexpr (n.op == OpKind::Lt) {
        truth[n.self] =
            operand_poisoned ? Tri::Unknown : detail_cont::tri_lt(a, b);
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::Le) {
        truth[n.self] =
            operand_poisoned ? Tri::Unknown : detail_cont::tri_le(a, b);
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::Gt) {
        truth[n.self] =
            operand_poisoned ? Tri::Unknown : detail_cont::tri_lt(b, a);
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::Ge) {
        truth[n.self] =
            operand_poisoned ? Tri::Unknown : detail_cont::tri_le(b, a);
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::Eq) {
        truth[n.self] =
            operand_poisoned ? Tri::Unknown : detail_cont::tri_eq(a, b);
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::Ne) {
        truth[n.self] = operand_poisoned
                            ? Tri::Unknown
                            : detail_cont::tri_not(detail_cont::tri_eq(a, b));
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::And) {
        truth[n.self] = detail_cont::tri_and(truth[n.a], truth[n.b]);
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::Or) {
        truth[n.self] = detail_cont::tri_or(truth[n.a], truth[n.b]);
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::Not) {
        truth[n.self] = detail_cont::tri_not(truth[n.a]);
        ranges[n.self] = {0.0, 1.0};

      } else if constexpr (n.op == OpKind::Select) {
        const Tri cond = truth[n.cond];
        // Only the live branch's poison counts.
        if (cond == Tri::True) {
          if (poisoned[n.a])
            return {false, static_cast<int>(n.self), OpKind::Select};
          ranges[n.self] = ranges[n.a];
          poisoned[n.self] = false;
        } else if (cond == Tri::False) {
          if (poisoned[n.b])
            return {false, static_cast<int>(n.self), OpKind::Select};
          ranges[n.self] = ranges[n.b];
          poisoned[n.self] = false;
        } else
          // May jump. Incomplete: branches agreeing there are still reported.
          return {false, static_cast<int>(n.self), OpKind::Select};

      } else {
        // Tensor ops or any unrecognised op: not supported yet.
        return {false, static_cast<int>(n.self), n.op};
      }
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
