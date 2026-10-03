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
//   2. For each comparison whose operands depend on the target, solve
//      lhs - rhs = 0 for the target. Both sides must be affine in it (built
//      with + -, unary - and * / by target-free values); anything else is a
//      compile-time error rather than a guess.
//   3. At each root, evaluate the whole function with the comparisons that
//      flip there forced to their outcome just above, and just below, the
//      root. The difference is the jump; roots with no jump are dropped.
//   4. Return as a static array (assuming finite discontinuities)
//
// The consteval entry points evaluate the function at compile time, using
// cx_std for exp/log/sqrt/erfc. sin/cos have no constexpr version, so a
// function that evaluates them needs the runtime (_rt) entry point.

#include "../autograd.h"
#include "../cx_std/cx_erfc.hpp"
#include "../cx_std/cx_exp.hpp"
#include "../cx_std/cx_log.hpp"
#include "../cx_std/cx_sqrt.hpp"
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

// The comparisons whose outcome flips as the target crosses a point. `==` and
// `!=` are left out: they take their other outcome only at the point itself,
// so the function's left and right limits there are equal and there is no
// jump (a single point carries no probability mass; see #67).
constexpr bool is_ordering(OpKind op) {
  return op == OpKind::Lt || op == OpKind::Le || op == OpKind::Gt ||
         op == OpKind::Ge;
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

template <info Fn> consteval std::size_t input_count() {
  std::size_t count = 0;
  for (const Node &n : nodes_of<Fn>)
    count += n.op == OpKind::Input;
  return count;
}

// How a node's value depends on the target input.
struct Dependence {
  bool varies = false; // it changes with the target
  bool affine = true;  // ... and only as c0 + c1 * target
};

// Per node, built in one forward pass (operands precede their users). Affine
// means built from the target with + -, unary - and * / by target-free
// values: what a comparison's operands must be for its crossing point to be
// solved exactly.
template <info Fn, std::size_t Target>
consteval std::vector<Dependence> dependence() {
  const auto nodes = nodes_of<Fn>;
  std::vector<Dependence> dep(nodes.size());
  for (const Node &n : nodes) {
    Dependence &d = dep[n.self];
    if (n.op == OpKind::Input) {
      d.varies = n.self == Target;
      continue;
    }
    const Dependence a = op_has_a(n.op) ? dep[n.a] : Dependence{};
    const Dependence b = op_has_b(n.op) ? dep[n.b] : Dependence{};
    d.varies =
        a.varies || b.varies || (op_has_cond(n.op) && dep[n.cond].varies);
    if (n.op == OpKind::Add || n.op == OpKind::Sub)
      d.affine = a.affine && b.affine;
    else if (n.op == OpKind::Neg || n.op == OpKind::Output)
      d.affine = a.affine;
    else if (n.op == OpKind::Mul)
      d.affine = a.affine && b.affine && !(a.varies && b.varies);
    else if (n.op == OpKind::Div)
      d.affine = a.affine && !b.varies;
    else
      d.affine = !d.varies;
  }
  return dep;
}

template <info Fn, std::size_t Target>
inline constexpr auto dependence_of =
    std::define_static_array(dependence<Fn, Target>());

// A comparison whose outcome changes as the target moves. `==` / `!=` count:
// they never make a point of their own, but where one shares a root with an
// ordering comparison it must be held at its limit (see analyze).
template <info Fn, std::size_t Target>
consteval bool is_crossing(const Node &n) {
  const auto dep = dependence_of<Fn, Target>;
  return is_comparison(n.op) && (dep[n.a].varies || dep[n.b].varies);
}

// The nodes a crossing comparison reads, directly or not: what has to be
// evaluated to place it.
template <info Fn, std::size_t Target>
consteval std::vector<char> crossing_operands() {
  const auto nodes = nodes_of<Fn>;
  std::vector<char> needed(nodes.size(), 0);
  for (std::size_t i = nodes.size(); i-- > 0;) {
    const Node &n = nodes[i];
    if (!needed[i] && !is_crossing<Fn, Target>(n))
      continue;
    if (op_has_a(n.op))
      needed[n.a] = 1;
    if (op_has_b(n.op))
      needed[n.b] = 1;
    if (op_has_cond(n.op))
      needed[n.cond] = 1;
  }
  return needed;
}

template <info Fn, std::size_t Target>
inline constexpr auto crossing_operands_of =
    std::define_static_array(crossing_operands<Fn, Target>());

// primal()'s arithmetic here. <cmath> is not constexpr in this toolchain, so
// during constant evaluation the cx_std versions stand in. There is no cx_std
// sin/cos: a sin or cos on an evaluated path only works through the runtime
// entry point.
struct CxMath : BuiltinMath {
  template <OpKind Op> static constexpr double unary(double x) {
    if consteval {
      if constexpr (Op == OpKind::Exp)
        return cx::exp(x);
      else if constexpr (Op == OpKind::Log)
        return cx::log(x);
      else if constexpr (Op == OpKind::Sqrt)
        return cx::sqrt(x);
      else if constexpr (Op == OpKind::Erfc)
        return cx::erfc(x);
      else if constexpr (Op == OpKind::Sin)
        return std::sin(x);
      else
        return std::cos(x);
    } else {
      if constexpr (Op == OpKind::Exp)
        return std::exp(x);
      else if constexpr (Op == OpKind::Log)
        return std::log(x);
      else if constexpr (Op == OpKind::Sqrt)
        return std::sqrt(x);
      else if constexpr (Op == OpKind::Erfc)
        return std::erfc(x);
      else if constexpr (Op == OpKind::Sin)
        return std::sin(x);
      else
        return std::cos(x);
    }
  }
};

// Value of node I from the values of the nodes before it, with the target
// input at x. Only plain copies of `n`'s fields appear in runtime code: naming
// the (consteval-only) Node there would make this function immediate.
template <info Fn, std::size_t I, std::size_t Target, std::size_t N,
          std::size_t NumArgs>
constexpr double node_value(const std::array<double, N> &val,
                            const std::array<double, NumArgs> &in, double x) {
  constexpr Node n = nodes_of<Fn>[I];
  constexpr OpKind op = n.op;
  constexpr std::size_t a = n.a, b = n.b, c = n.cond;
  if constexpr (op == OpKind::Input) {
    if constexpr (I == Target)
      return x;
    else
      return in[I];
  } else if constexpr (op == OpKind::Const) {
    return static_cast<double>([:n.leaf:]);
  } else if constexpr (op_has_primal(op)) {
    return primal<op, CxMath>(val[a], val[b], val[c]);
  } else {
    static_assert(false, "discontinuity_analysis: unsupported operation");
  }
}

// d(node I)/d(target). Every target-dependent node a crossing reads is affine
// (see Dependence), so only those ops need a rule.
template <info Fn, std::size_t I, std::size_t Target, std::size_t N>
constexpr double node_tangent(const std::array<double, N> &val,
                              const std::array<double, N> &tan) {
  constexpr Node n = nodes_of<Fn>[I];
  constexpr OpKind op = n.op;
  constexpr std::size_t a = n.a, b = n.b;
  if constexpr (!dependence_of<Fn, Target>[I].varies)
    return 0.0;
  else if constexpr (op == OpKind::Input)
    return 1.0;
  else if constexpr (op == OpKind::Add)
    return tan[a] + tan[b];
  else if constexpr (op == OpKind::Sub)
    return tan[a] - tan[b];
  else if constexpr (op == OpKind::Neg)
    return -tan[a];
  else if constexpr (op == OpKind::Output)
    return tan[a];
  else if constexpr (op == OpKind::Mul)
    return tan[a] * val[b] + val[a] * tan[b];
  else if constexpr (op == OpKind::Div)
    return tan[a] / val[b];
  else
    return 0.0; // not affine: analyze rejects any crossing that reads it
}

template <std::size_t N> struct Sweep {
  std::array<double, N> val = {};
  std::array<double, N> tan = {}; // derivative in the target
};

// Values and target derivatives of every node a crossing comparison reads,
// with the target input at x. Guards are not consulted: a comparison's
// operands are evaluated even where its branch is not taken.
template <info Fn, std::size_t Target, std::size_t NumArgs>
constexpr auto operand_sweep(const std::array<double, NumArgs> &in, double x) {
  static constexpr auto nodes = nodes_of<Fn>;
  static constexpr auto needed = crossing_operands_of<Fn, Target>;
  Sweep<nodes.size()> s;
  template for (constexpr Node n : nodes) {
    if constexpr (needed[n.self]) {
      constexpr std::size_t i = n.self;
      s.val[i] = node_value<Fn, i, Target>(s.val, in, x);
      s.tan[i] = node_tangent<Fn, i, Target>(s.val, s.tan);
    }
  }
  return s;
}

// Fn's value with the target input at x, taking branches as Fn does (a
// guarded node is skipped when its guard is false), except that comparison i
// takes outcome forced[i] where that is 0 or 1.
template <info Fn, std::size_t Target, std::size_t N, std::size_t NumArgs>
constexpr double value_with(const std::array<double, NumArgs> &in, double x,
                            const std::array<signed char, N> &forced) {
  static constexpr auto nodes = nodes_of<Fn>;
  std::array<double, N> val = {};
  template for (constexpr Node n : nodes) {
    constexpr std::size_t i = n.self, guard = n.guard;
    if constexpr (guard != UNGUARDED) {
      if (val[guard] == 0.0)
        continue;
    }
    if constexpr (is_comparison(n.op)) {
      if (forced[i] >= 0) {
        val[i] = forced[i];
        continue;
      }
    }
    val[i] = node_value<Fn, i, Target>(val, in, x);
  }
  return val[N - 1];
}

// The analysis behind every entry point.
//
// Each comparison that varies with the target compares g = lhs - rhs with 0,
// and g is affine in the target, so its value and slope at 0 place the root
// exactly and say which outcome holds just above it. At each root of an
// ordering comparison, the jump is Fn's right limit minus its left limit:
// Fn evaluated at the root with every comparison rooted there forced to its
// outcome just above, minus the same with the outcome just below. Whatever
// sits between the comparison and the output (`!`, `&&`, `||`, nested
// selects, scaling, other jumps at the same point) is evaluated rather than
// pattern-matched. `==` / `!=` rooted there take their outcome off the point
// on both sides. Roots with no jump are not reported.
//
// `in` holds the function's arguments; the target's slot is ignored.
template <info Fn, std::size_t Target, std::size_t MaxPoints,
          std::size_t NumArgs>
constexpr DiscontinuityPointsWithAmplitudes<MaxPoints>
analyze(const std::array<double, NumArgs> &in) {
  static constexpr auto nodes = nodes_of<Fn>;
  static constexpr auto dep = dependence_of<Fn, Target>;
  static_assert(input_count<Fn>() == NumArgs,
                "discontinuity_analysis: pass one fixed value for each "
                "argument after the target");
  constexpr std::size_t N = nodes.size();

  // Per comparison slot: whether it flips at all, where, its outcome just
  // below and just above that point, and whether it can make a jump there.
  std::array<bool, N> rooted = {}, makes_point = {};
  std::array<double, N> root = {};
  std::array<signed char, N> below = {}, above = {};

  const auto at_zero = operand_sweep<Fn, Target>(in, 0.0);
  template for (constexpr Node n : nodes) {
    if constexpr (is_crossing<Fn, Target>(n)) {
      static_assert(dep[n.a].affine && dep[n.b].affine,
                    "discontinuity_analysis: both sides of a comparison on "
                    "the target must be affine in it (built from it with + - "
                    "and * / by values that do not depend on it)");
      constexpr std::size_t i = n.self, a = n.a, b = n.b;
      constexpr OpKind op = n.op;
      const double slope = at_zero.tan[a] - at_zero.tan[b];
      // A flat g never flips, e.g. `s * k > 1` at k = 0.
      if (slope != 0.0) {
        rooted[i] = true;
        makes_point[i] = is_ordering(op);
        // `+ 0.0` turns a -0 root into 0.
        root[i] = -(at_zero.val[a] - at_zero.val[b]) / slope + 0.0;
        // Just above the root, g has the sign of its slope.
        const bool g_positive = slope > 0.0;
        if constexpr (op == OpKind::Gt || op == OpKind::Ge) {
          above[i] = g_positive;
          below[i] = !g_positive;
        } else if constexpr (op == OpKind::Lt || op == OpKind::Le) {
          above[i] = !g_positive;
          below[i] = g_positive;
        } else {
          above[i] = below[i] = (op == OpKind::Ne);
        }
      }
    }
  }

  DiscontinuityCollectorWithAmplitudes<MaxPoints> collector;
  for (std::size_t c = 0; c < N; ++c) {
    if (!makes_point[c])
      continue;
    const double point = root[c];
    std::array<signed char, N> right, left;
    right.fill(-1);
    left.fill(-1);
    for (std::size_t j = 0; j < N; ++j) {
      if (rooted[j] && root[j] == point) {
        right[j] = above[j];
        left[j] = below[j];
      }
    }
    const double jump = value_with<Fn, Target>(in, point, right) -
                        value_with<Fn, Target>(in, point, left);
    if (jump != 0.0)
      collector.add_point_with_amplitude(point, jump);
  }

  collector.sort_by_points();
  return collector.get_result();
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
