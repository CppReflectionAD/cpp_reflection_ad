#ifndef DISCONTINUITY_ANALYSIS_HPP
#define DISCONTINUITY_ANALYSIS_HPP

// discontinuity_analysis.hpp — compile-time discontinuity point extraction.
//
// Given a function f(x₀, x₁, ..., xₙ) and a target input index i, with fixed
// values for all other inputs, finds all points where f has a discontinuity
// along the i-th argument.
//
// Usage:
//   double f(double x, double y) { return (x > y) ? 1.0 : 0.0; }
//   constexpr auto points = ad::get_discontinuity_points<^^f, 0>(2.0);
//   // points = [2.0] (discontinuity at x = y = 2.0)
//
// Strategy:
//   1. Build the DAG via reflection (like autograd.h)
//   2. Traverse to find all comparison/select operations
//   3. For each comparison involving the target input, extract the boundary
//      value being compared against
//   4. Return as a static array (assuming finite discontinuities)

#include "../autograd.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <meta>

namespace ad {

using std::meta::info;

// ---------------------------------------------------------------------------
// Helper: extract discontinuity points from comparisons
// ---------------------------------------------------------------------------
namespace detail_disc {

// First: helper to build a consteval-friendly std::array from variadic args
template <typename T, typename... Args>
consteval std::array<T, sizeof...(Args)> make_array_impl(Args... args) {
  return {static_cast<T>(args)...};
}

constexpr double signum(double x) {
  return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}

constexpr double comparison_jump_sign(OpKind op, double slope) {
  const double slope_sign = signum(slope);
  if (slope_sign == 0.0)
    return 0.0;
  if (op == OpKind::Lt || op == OpKind::Le)
    return -slope_sign;
  if (op == OpKind::Gt || op == OpKind::Ge)
    return slope_sign;
  return 1.0;
}

} // namespace detail_disc

// ---------------------------------------------------------------------------
// Main API: get_discontinuity_points<Fn, TargetArgIndex>(fixed_args...)
//
// Returns a static array of discontinuity points along TargetArgIndex
// while other arguments are fixed at the provided values.
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
// Stores pairs: (point, amplitude), flattened into single array
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
};

// Helper to collect discontinuity points and amplitudes during DAG traversal.
// constexpr (not consteval) so the runtime entry point can use it too.
template <std::size_t MaxPoints = 16>
struct DiscontinuityCollectorWithAmplitudes {
  std::array<double, MaxPoints * 2> data = {}; // pairs of (point, amplitude)
  std::size_t count = 0;

  constexpr void add_point_with_amplitude(double point, double amplitude) {
    if (count < MaxPoints) {
      // Check for duplicate points
      bool found = false;
      for (std::size_t i = 0; i < count; ++i) {
        if (data[i * 2] == point) {
          found = true;
          break;
        }
      }
      if (!found) {
        data[count * 2] = point;
        data[count * 2 + 1] = amplitude;
        count++;
      }
    }
  }

  constexpr void sort_by_points() {
    // Bubble sort by points (consteval-friendly)
    for (std::size_t i = 0; i < count; ++i) {
      for (std::size_t j = i + 1; j < count; ++j) {
        if (data[j * 2] < data[i * 2]) {
          // Swap points
          double tmp_p = data[i * 2];
          data[i * 2] = data[j * 2];
          data[j * 2] = tmp_p;
          // Swap amplitudes
          double tmp_a = data[i * 2 + 1];
          data[i * 2 + 1] = data[j * 2 + 1];
          data[j * 2 + 1] = tmp_a;
        }
      }
    }
  }

  // Return result with count information
  constexpr DiscontinuityPointsWithAmplitudes<MaxPoints> get_result() const {
    return {data, count};
  }
};

// ---------------------------------------------------------------------------
// The analysis. Every entry point below calls detail_disc::analyze, so a fix
// to how comparisons, points or amplitudes are read lands in one place.
// ---------------------------------------------------------------------------
namespace detail_disc {

// The reflected DAG of Fn, built once and shared by every analysis of Fn.
template <info Fn>
inline constexpr auto nodes_of = std::define_static_array(build_nodes<Fn>());

constexpr bool is_comparison(OpKind op) {
  return op == OpKind::Lt || op == OpKind::Le || op == OpKind::Gt ||
         op == OpKind::Ge || op == OpKind::Eq || op == OpKind::Ne;
}

constexpr bool is_arithmetic(OpKind op) {
  return op == OpKind::Add || op == OpKind::Sub || op == OpKind::Mul ||
         op == OpKind::Div;
}

// The function's arguments: the fixed values follow TargetArgIndex, and every
// other slot (the target's included) is 0.
template <std::size_t TargetArgIndex, typename... FixedArgs>
constexpr std::array<double, TargetArgIndex + 1 + sizeof...(FixedArgs)>
make_inputs(FixedArgs... fixed_args) {
  std::array<double, TargetArgIndex + 1 + sizeof...(FixedArgs)> in = {};
  std::size_t idx = 0;
  ((in[TargetArgIndex + 1 + idx++] = static_cast<double>(fixed_args)), ...);
  return in;
}

// Value of node I with the target input at x, for the shapes the analyzer
// understands: literals, inputs, `Levels` nested levels of + - * /, and (where
// AllowNeg) a negated literal. Anything else reads as 0.
template <info Fn, std::size_t I, std::size_t Target, std::size_t Levels,
          bool AllowNeg, std::size_t NumArgs>
constexpr double shallow_value(const std::array<double, NumArgs> &in,
                               double x) {
  static constexpr auto nodes = nodes_of<Fn>;
  constexpr Node n = nodes[I];
  // Naming `n` (a consteval-only Node) in a runtime expression would make this
  // function immediate, so runtime code reads a plain copy of the slot.
  constexpr std::size_t slot = n.self;
  if constexpr (n.op == OpKind::Const) {
    return static_cast<double>([:n.leaf:]);
  } else if constexpr (n.op == OpKind::Input) {
    if constexpr (slot == Target)
      return x;
    else if constexpr (slot < NumArgs)
      return in[slot];
    else
      return 0.0;
  } else if constexpr (AllowNeg && n.op == OpKind::Neg &&
                       nodes[n.a].op == OpKind::Const) {
    constexpr Node operand = nodes[n.a];
    return -static_cast<double>([:operand.leaf:]);
  } else if constexpr (Levels > 0 && is_arithmetic(n.op)) {
    const double l = shallow_value<Fn, n.a, Target, Levels - 1, false>(in, x);
    const double r = shallow_value<Fn, n.b, Target, Levels - 1, false>(in, x);
    if constexpr (n.op == OpKind::Add)
      return l + r;
    else if constexpr (n.op == OpKind::Sub)
      return l - r;
    else if constexpr (n.op == OpKind::Mul)
      return l * r;
    else
      return (r != 0.0) ? (l / r) : 0.0;
  } else {
    return 0.0;
  }
}

template <info Fn, std::size_t I, std::size_t Target>
consteval bool is_target_input() {
  constexpr Node n = nodes_of<Fn>[I];
  return n.op == OpKind::Input && n.self == Target;
}

// An operand the boundary can be read from: a literal, another input, or one
// + - * / node.
template <info Fn, std::size_t I, std::size_t Target, std::size_t NumArgs>
consteval bool is_boundary_operand() {
  constexpr Node n = nodes_of<Fn>[I];
  return n.op == OpKind::Const ||
         (n.op == OpKind::Input && n.self != Target && n.self < NumArgs) ||
         is_arithmetic(n.op);
}

// Slope of (lhs - rhs) in the target for comparison C, or 0 when C does not
// have one of the six recognised shapes:
//   Case 1/2: the target against a constant (target on the left/right);
//   Case 3/4: the target against another input;
//   Case 5/6: the target against a + - * / node, e.g. `strike - 1`.
template <info Fn, std::size_t C, std::size_t Target, std::size_t NumArgs>
consteval double boundary_slope() {
  constexpr Node n = nodes_of<Fn>[C];
  if constexpr (is_target_input<Fn, n.a, Target>() &&
                is_boundary_operand<Fn, n.b, Target, NumArgs>())
    return 1.0;
  else if constexpr (is_target_input<Fn, n.b, Target>() &&
                     is_boundary_operand<Fn, n.a, Target, NumArgs>())
    return -1.0;
  else
    return 0.0;
}

// The value the target is compared against in comparison C: its other
// operand, read with the target slot at 0.
template <info Fn, std::size_t C, std::size_t Target, std::size_t NumArgs>
constexpr double boundary_point(const std::array<double, NumArgs> &in) {
  constexpr Node n = nodes_of<Fn>[C];
  constexpr std::size_t other =
      boundary_slope<Fn, C, Target, NumArgs>() > 0.0 ? n.b : n.a;
  return shallow_value<Fn, other, Target, 1, false>(in, 0.0);
}

// Sign the output picks up from Select S's direct parents: -1 if S is negated
// or is the right operand of a Sub.
template <info Fn, std::size_t S> consteval double parent_scale() {
  for (const Node &p : nodes_of<Fn>) {
    if ((p.op == OpKind::Neg && p.a == S) || (p.op == OpKind::Sub && p.b == S))
      return -1.0;
  }
  return 1.0;
}

// The analysis behind every entry point. For each comparison involving the
// target, find its boundary; for each Select it drives, the jump there.
// PointsOnly records every boundary, with amplitude 0, whether or not a Select
// uses it. `in` holds the function's arguments; the target's slot is ignored.
template <info Fn, std::size_t Target, std::size_t MaxPoints, bool PointsOnly,
          std::size_t NumArgs>
constexpr DiscontinuityPointsWithAmplitudes<MaxPoints>
analyze(const std::array<double, NumArgs> &in) {
  static constexpr auto nodes = nodes_of<Fn>;
  DiscontinuityCollectorWithAmplitudes<MaxPoints> collector;

  template for (constexpr Node n : nodes) {
    if constexpr (is_comparison(n.op)) {
      constexpr double slope = boundary_slope<Fn, n.self, Target, NumArgs>();
      if constexpr (slope != 0.0) {
        const double point = boundary_point<Fn, n.self, Target>(in);
        if constexpr (PointsOnly) {
          collector.add_point_with_amplitude(point, 0.0);
        } else {
          template for (constexpr Node s : nodes) {
            if constexpr (s.op == OpKind::Select && s.cond == n.self) {
              constexpr double sign = comparison_jump_sign(n.op, slope) *
                                      parent_scale<Fn, s.self>();
              const double jump =
                  shallow_value<Fn, s.a, Target, 2, true>(in, point) -
                  shallow_value<Fn, s.b, Target, 2, true>(in, point);
              collector.add_point_with_amplitude(point, jump * sign);
            }
          }
        }
      }
    }
  }

  collector.sort_by_points();
  return collector.get_result();
}

} // namespace detail_disc

template <info Fn, std::size_t TargetArgIndex, std::size_t MaxPoints = 16,
          typename... FixedArgs>
consteval DiscontinuityPoints<MaxPoints>
get_discontinuity_points(FixedArgs... fixed_args) {
  const auto found = detail_disc::analyze<Fn, TargetArgIndex, MaxPoints, true>(
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
  return detail_disc::analyze<Fn, TargetArgIndex, MaxPoints, false>(
      detail_disc::make_inputs<TargetArgIndex>(fixed_args...));
}

// Runtime version: accepts dynamic (non-constexpr) fixed arguments. The DAG is
// still analysed at compile time; only the arithmetic on the argument values
// runs at runtime.
template <info Fn, std::size_t TargetArgIndex, std::size_t MaxPoints = 16,
          typename... FixedArgs>
inline DiscontinuityPointsWithAmplitudes<MaxPoints>
get_discontinuity_points_and_amplitudes_rt(FixedArgs... fixed_args) {
  return detail_disc::analyze<Fn, TargetArgIndex, MaxPoints, false>(
      detail_disc::make_inputs<TargetArgIndex>(fixed_args...));
}

} // namespace ad

#endif // DISCONTINUITY_ANALYSIS_HPP
