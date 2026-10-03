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
//   3. At each root, evaluate the whole function with the crossings there
//      forced to their outcome just above, and just below, the root, and
//      those rooted a few ulps away held at their side of it. The difference
//      is the jump. Roots with no jump are dropped, and so are those whose
//      jump is only rounding: one that vanishes or changes sign a few ulps
//      away, as at a kink whose root is not a double.
//   4. Return as a static array (assuming finite discontinuities)
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
// k = inf, which never flips), or to have more points than MaxPoints.

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

// Collects (point, amplitude) pairs during the analysis. constexpr (not
// consteval) so the runtime entry point can use it too.
template <std::size_t MaxPoints = 16>
struct DiscontinuityCollectorWithAmplitudes {
  std::array<double, MaxPoints * 2> data = {}; // pairs of (point, amplitude)
  std::size_t count = 0;

  // A point that does not fit is an error, not a silent omission.
  constexpr void add_point_with_amplitude(double point, double amplitude) {
    if (count == MaxPoints)
      throw "discontinuity_analysis: Fn has more discontinuities than "
            "MaxPoints; pass a larger MaxPoints";
    data[count * 2] = point;
    data[count * 2 + 1] = amplitude;
    count++;
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

template <info Fn> consteval std::size_t input_count() {
  std::size_t count = 0;
  for (const Node &n : nodes_of<Fn>)
    count += n.op == OpKind::Input;
  return count;
}

// How a node's value depends on the target input.
struct Dependence {
  bool varies = false;  // it changes with the target
  bool affine = true;   // ... and only as c0 + c1 * target
  bool stepwise = true; // ... and only in steps, where a comparison flips
};

// A value that changes with the target other than in steps: one whose own
// zeros, or whose comparisons, have to be solved for.
constexpr bool varies_continuously(Dependence d) {
  return d.varies && !d.stepwise;
}

// Per node, built in one forward pass (operands precede their users). Affine
// means built from the target with + -, unary - and * / by target-free
// values, and `?:` on a target-free condition: what a crossing's sides must be
// for its point to be solved exactly.
// Stepwise means piecewise constant: every comparison or logical op is, and
// so is anything built only from stepwise and target-free values.
template <info Fn, std::size_t Target>
consteval std::vector<Dependence> dependence() {
  const auto nodes = nodes_of<Fn>;
  std::vector<Dependence> dep(nodes.size());
  for (const Node &n : nodes) {
    Dependence &d = dep[n.self];
    if (n.op == OpKind::Input) {
      d.varies = n.self == Target;
      d.stepwise = !d.varies;
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
    else if (n.op == OpKind::Select && !dep[n.cond].varies)
      d.affine = a.affine && b.affine; // one branch, whatever the target
    else
      d.affine = !d.varies;
    // A Select's condition only picks a branch: if both branches are
    // stepwise, so is the Select.
    d.stepwise = op_is_boolean(n.op) ||
                 (!varies_continuously(a) && !varies_continuously(b));
  }
  return dep;
}

template <info Fn, std::size_t Target>
inline constexpr auto dependence_of =
    std::define_static_array(dependence<Fn, Target>());

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
  const auto dep = dependence_of<Fn, Target>;
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

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

constexpr bool is_finite(double x) { return x == x && x != kInf && x != -kInf; }

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
    return primal<op, IeeeCxMath>(val[a], val[b], val[c]);
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
  constexpr auto dep = dependence_of<Fn, Target>;
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
  static constexpr auto dep = dependence_of<Fn, Target>;
  static constexpr auto cone = crossing_cone_of<Fn, Target>;
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

// The nodes crossing I's gap reads, directly or not, in order: its sides (or
// the condition itself) and their operands. Not their guards: whether a node
// is reached does not depend on the target, so any sweep already says.
template <info Fn, std::size_t I>
consteval std::vector<std::size_t> gap_cone() {
  const auto nodes = nodes_of<Fn>;
  std::vector<char> read(I + 1, 0);
  if (is_comparison(nodes[I].op))
    read[nodes[I].a] = read[nodes[I].b] = 1;
  else
    read[I] = 1;
  for (std::size_t i = I + 1; i-- > 0;) {
    const Node &n = nodes[i];
    if (!read[i])
      continue;
    if (op_has_a(n.op))
      read[n.a] = 1;
    if (op_has_b(n.op))
      read[n.b] = 1;
    if (op_has_cond(n.op))
      read[n.cond] = 1;
  }
  std::vector<std::size_t> cone;
  for (std::size_t i = 0; i <= I; ++i)
    if (read[i])
      cone.push_back(i);
  return cone;
}

template <info Fn, std::size_t I>
inline constexpr auto gap_cone_of = std::define_static_array(gap_cone<Fn, I>());

// Crossing I's gap (as gap() reads it from a sweep) with the target at x,
// evaluating only the nodes it reads. A node is skipped, as sweep() skips it,
// if its guard can never hold; `reach` is any sweep, since that does not
// depend on x.
template <info Fn, std::size_t Target, std::size_t I, std::size_t N,
          std::size_t NumArgs>
constexpr double gap_at(const std::array<double, NumArgs> &in, double x,
                        const Sweep<N> &reach) {
  static constexpr auto nodes = nodes_of<Fn>;
  std::array<double, N> val = {};
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
    return IeeeCxMath::sub(val[a], val[b]);
  else
    return val[I];
}

// x moved k ulps up (k > 0) or down (k < 0), through ±0.
constexpr double step_ulps(double x, int k) {
  const auto up = [](double v) {
    if (v == 0.0)
      return std::numeric_limits<double>::denorm_min();
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(v);
    return std::bit_cast<double>(v > 0.0 ? bits + 1 : bits - 1);
  };
  for (; k > 0; --k)
    x = up(x);
  for (; k < 0; ++k)
    x = -up(-x);
  return x;
}

// How far a root is moved to where crossing I's sides are equal (see snap).
inline constexpr int kSnapUlps = 4;
// How near another crossing's root must be to a point to be held at its side
// of it (see analyze).
inline constexpr int kHoldUlps = 4;
// How far either side of a point a jump must keep its sign to be more than
// rounding (see analyze).
inline constexpr int kKinkUlps = 4;

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
template <info Fn, std::size_t Target, std::size_t N, std::size_t NumArgs>
constexpr double value_with(const std::array<double, NumArgs> &in, double x,
                            const std::array<signed char, N> &forced) {
  static constexpr auto nodes = nodes_of<Fn>;
  std::array<double, N> val = {};
  // Node j read as a condition: its forced truth, else its value (nonzero is
  // true).
  const auto cond = [&](std::size_t j) {
    return forced[j] >= 0 ? static_cast<double>(forced[j]) : val[j];
  };
  template for (constexpr Node n : nodes) {
    constexpr std::size_t i = n.self, guard = n.guard, a = n.a, b = n.b,
                          c = n.cond;
    constexpr OpKind op = n.op;
    if constexpr (guard != UNGUARDED) {
      if (cond(guard) == 0.0)
        continue;
    }
    if constexpr (is_comparison(op)) {
      if (forced[i] >= 0) {
        val[i] = forced[i];
        continue;
      }
    }
    if constexpr (op == OpKind::Not || op == OpKind::And || op == OpKind::Or)
      val[i] = primal<op, IeeeCxMath>(cond(a), cond(b), 0.0);
    else if constexpr (op == OpKind::Select)
      val[i] = primal<op, IeeeCxMath>(val[a], val[b], cond(c));
    else
      val[i] = node_value<Fn, i, Target>(val, in, x);
  }
  return val[N - 1];
}

// The analysis behind every entry point.
//
// Each crossing compares g with 0 (g = lhs - rhs, or the condition's value),
// and g is affine in the target, so its value and slope at 0 place the root
// exactly and say which outcome holds just above it. At each root of an
// ordering comparison, the jump is Fn's right limit minus its left limit:
// Fn evaluated at the root with every crossing rooted there forced to its
// outcome just above, minus the same with the outcome just below. Whatever
// sits between the comparison and the output (`!`, `&&`, `||`, nested
// selects, scaling, other jumps at the same point) is evaluated rather than
// pattern-matched. `==` / `!=` and conditions rooted there take their outcome
// off the point on both sides, and crossings rooted a few ulps away the
// outcome on their side of it. Roots with no jump, or with one that vanishes
// or changes sign a few ulps away (rounding at a kink), are not reported.
//
// `in` holds the function's arguments; the target's slot is ignored.
template <info Fn, std::size_t Target, std::size_t MaxPoints,
          std::size_t NumArgs>
constexpr DiscontinuityPointsWithAmplitudes<MaxPoints>
analyze(const std::array<double, NumArgs> &in) {
  static constexpr auto nodes = nodes_of<Fn>;
  static constexpr auto dep = dependence_of<Fn, Target>;
  static constexpr auto crossing = crossings_of<Fn, Target>;
  static_assert(Target < input_count<Fn>(),
                "discontinuity_analysis: the target index is not an argument "
                "of the function");
  static_assert(input_count<Fn>() == NumArgs,
                "discontinuity_analysis: pass one fixed value for each "
                "argument except the target, in order");
  constexpr std::size_t N = nodes.size();

  // Per crossing slot: whether it flips at all, where, its outcome just
  // below and just above that point, and whether it can make a jump there.
  std::array<bool, N> rooted = {}, makes_point = {};
  std::array<double, N> root = {};
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
      // No root if Fn never reaches the crossing, if g is flat (`s * k > 1`
      // at k = 0) or, with every coefficient finite, its constant is not
      // (`s > 1 / k` at k = 0: -inf for every target, never true), or if the
      // root is beyond the doubles.
      const double r =
          at_zero.reached(i) && slope != 0.0 && is_finite(g0)
              ? -(g0 / slope) + 0.0 // `+ 0.0` turns a -0 root into 0
              : kNaN;
      if (is_finite(r)) {
        rooted[i] = true;
        makes_point[i] = is_ordering(op);
        root[i] = snap<Fn, Target, i>(in, r, at_zero);
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

  DiscontinuityCollectorWithAmplitudes<MaxPoints> collector;
  // Crossings whose root has been measured, so each point is measured once.
  std::array<bool, N> measured = {};
  for (std::size_t c = 0; c < N; ++c) {
    if (!makes_point[c] || measured[c])
      continue;
    const double point = root[c];
    // Crossings rooted at the point take their outcome above it on the
    // right and below it on the left. Those rooted within kHoldUlps of it
    // take the outcome on their side of it on both, by where their roots
    // lie rather than by Fn's arithmetic: that can make a comparison's sides
    // exactly equal on a run of doubles, so a crossing rooted just below the
    // point may still read false there. Such crossings are often one
    // crossing in exact arithmetic (`s * 1.1 > k && s / 0.9 > k / 0.99`),
    // whose jump would otherwise be lost at both roots.
    const double lo = step_ulps(point, -kHoldUlps),
                 hi = step_ulps(point, kHoldUlps);
    std::array<signed char, N> right, left;
    right.fill(-1);
    left.fill(-1);
    for (std::size_t j = 0; j < N; ++j) {
      if (!rooted[j] || root[j] < lo || hi < root[j])
        continue;
      if (root[j] == point) {
        right[j] = above[j];
        left[j] = below[j];
        measured[j] = true;
      } else {
        right[j] = left[j] = root[j] < point ? above[j] : below[j];
      }
    }
    const double from_right = value_with<Fn, Target>(in, point, right);
    const double from_left = value_with<Fn, Target>(in, point, left);
    if (!is_finite(from_right) || !is_finite(from_left))
      throw "discontinuity_analysis: Fn is not finite on one side of a "
            "crossing point (a pole or an undefined branch there, not a "
            "jump)";
    const double jump = from_right - from_left;
    // A kink -- continuous where its branches meet -- has no jump at its
    // exact root, but that need not be a double, nor the point (snap finds
    // a double where the crossing's sides are equal, not the branches'), so
    // rounding can leave a tiny one here. Measured kKinkUlps either side,
    // with the same crossings forced, it then vanishes or changes sign; a
    // jump keeps its sign. A side that is not finite says nothing (`s > k ?
    // sqrt(s - k) + 1 : 0` is NaN just below k with the right side forced),
    // so the difference must not stop constant evaluation either.
    const auto jump_at = [&](double x) {
      return IeeeCxMath::sub(value_with<Fn, Target>(in, x, right),
                             value_with<Fn, Target>(in, x, left));
    };
    const auto disagrees = [&](double other) {
      return is_finite(other) &&
             (other == 0.0 || (other > 0.0) != (jump > 0.0));
    };
    if (jump != 0.0 && !disagrees(jump_at(step_ulps(point, -kKinkUlps))) &&
        !disagrees(jump_at(step_ulps(point, kKinkUlps))))
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
