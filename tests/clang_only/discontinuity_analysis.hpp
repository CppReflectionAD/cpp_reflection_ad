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

// Helper to collect discontinuity points and amplitudes during DAG traversal
template <std::size_t MaxPoints = 16>
struct DiscontinuityCollectorWithAmplitudes {
  std::array<double, MaxPoints * 2> data = {}; // pairs of (point, amplitude)
  std::size_t count = 0;

  consteval void add_point_with_amplitude(double point, double amplitude) {
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

  consteval void sort_by_points() {
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
  consteval DiscontinuityPointsWithAmplitudes<MaxPoints> get_result() const {
    return {data, count};
  }
};

// Helper to collect discontinuity points during DAG traversal
template <std::size_t MaxPoints = 16> struct DiscontinuityCollector {
  std::array<double, MaxPoints> points = {}; // Initialize all to 0.0
  std::size_t count = 0;

  consteval void add_point(double v) {
    if (count < MaxPoints) {
      // Check for duplicates
      bool found = false;
      for (std::size_t i = 0; i < count; ++i) {
        if (points[i] == v) {
          found = true;
          break;
        }
      }
      if (!found) {
        points[count++] = v;
      }
    }
  }

  consteval void sort_points() {
    // Bubble sort (consteval-friendly)
    for (std::size_t i = 0; i < count; ++i) {
      for (std::size_t j = i + 1; j < count; ++j) {
        if (points[j] < points[i]) {
          double tmp = points[i];
          points[i] = points[j];
          points[j] = tmp;
        }
      }
    }
  }

  // Return result with count information
  consteval DiscontinuityPoints<MaxPoints> get_result() const {
    return {points, count};
  }
};

// Core analysis: walk the DAG, find comparisons involving target input,
// extract their boundary values. All logic is inlined for constexpr
// compatibility.
template <info Fn, std::size_t TargetArgIndex, std::size_t NumArgs,
          std::size_t MaxPoints = 16>
consteval DiscontinuityCollector<MaxPoints>
analyze_discontinuities_impl(const std::array<double, NumArgs> &fixed_args) {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());

  DiscontinuityCollector<MaxPoints> collector;

  // Iterate over all nodes looking for comparisons
  template for (constexpr auto n : nodes) {
    // Check if this node is a comparison operation
    if constexpr (n.op == OpKind::Lt || n.op == OpKind::Le ||
                  n.op == OpKind::Gt || n.op == OpKind::Ge ||
                  n.op == OpKind::Eq || n.op == OpKind::Ne) {
      if constexpr (n.a < nodes.size() && n.b < nodes.size()) {
        constexpr auto operand_a = nodes[n.a];
        constexpr auto operand_b = nodes[n.b];

        // Case 1: a is the target input, b is a constant or non-target input
        if constexpr (operand_a.op == OpKind::Input &&
                      operand_a.self == TargetArgIndex &&
                      operand_b.op == OpKind::Const) {
          // Extract the constant value from b
          constexpr double b_const = static_cast<double>([:operand_b.leaf:]);
          collector.add_point(b_const);
        }

        // Case 2: b is the target input, a is a constant or non-target input
        if constexpr (operand_b.op == OpKind::Input &&
                      operand_b.self == TargetArgIndex &&
                      operand_a.op == OpKind::Const) {
          // Extract the constant value from a
          constexpr double a_const = static_cast<double>([:operand_a.leaf:]);
          collector.add_point(a_const);
        }

        // Case 3: a is target input, b is another input (non-target)
        if constexpr (operand_a.op == OpKind::Input &&
                      operand_a.self == TargetArgIndex &&
                      operand_b.op == OpKind::Input &&
                      operand_b.self != TargetArgIndex &&
                      operand_b.self < NumArgs) {
          // Use the fixed value of operand_b
          double b_val = fixed_args[operand_b.self];
          collector.add_point(b_val);
        }

        // Case 4: b is target input, a is another input (non-target)
        if constexpr (operand_b.op == OpKind::Input &&
                      operand_b.self == TargetArgIndex &&
                      operand_a.op == OpKind::Input &&
                      operand_a.self != TargetArgIndex &&
                      operand_a.self < NumArgs) {
          // Use the fixed value of operand_a
          double a_val = fixed_args[operand_a.self];
          collector.add_point(a_val);
        }
      }
    }
  }

  collector.sort_points();
  return collector;
}

// Core analysis: extract both discontinuity points and amplitudes
template <info Fn, std::size_t TargetArgIndex, std::size_t NumArgs,
          std::size_t MaxPoints = 16>
consteval DiscontinuityCollectorWithAmplitudes<MaxPoints>
analyze_discontinuities_with_amplitudes_impl(
    const std::array<double, NumArgs> &fixed_args) {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());

  DiscontinuityCollectorWithAmplitudes<MaxPoints> collector;

  // Iterate over all nodes looking for comparisons
  template for (constexpr auto n : nodes) {
    // Check if this node is a comparison operation
    if constexpr (n.op == OpKind::Lt || n.op == OpKind::Le ||
                  n.op == OpKind::Gt || n.op == OpKind::Ge ||
                  n.op == OpKind::Eq || n.op == OpKind::Ne) {
      if constexpr (n.a < nodes.size() && n.b < nodes.size()) {
        constexpr auto operand_a = nodes[n.a];
        constexpr auto operand_b = nodes[n.b];

        // Determine if this comparison involves the target input
        bool involves_target = false;
        double point_value = 0.0;

        // Case 1: a is the target input, b is a constant
        if constexpr (operand_a.op == OpKind::Input &&
                      operand_a.self == TargetArgIndex &&
                      operand_b.op == OpKind::Const) {
          involves_target = true;
          constexpr double b_const = static_cast<double>([:operand_b.leaf:]);
          point_value = b_const;
        }

        // Case 2: b is the target input, a is a constant
        else if constexpr (operand_b.op == OpKind::Input &&
                           operand_b.self == TargetArgIndex &&
                           operand_a.op == OpKind::Const) {
          involves_target = true;
          constexpr double a_const = static_cast<double>([:operand_a.leaf:]);
          point_value = a_const;
        }

        // Case 3: a is target input, b is another input (non-target)
        else if constexpr (operand_a.op == OpKind::Input &&
                           operand_a.self == TargetArgIndex &&
                           operand_b.op == OpKind::Input &&
                           operand_b.self != TargetArgIndex &&
                           operand_b.self < NumArgs) {
          involves_target = true;
          point_value = fixed_args[operand_b.self];
        }

        // Case 4: b is target input, a is another input (non-target)
        else if constexpr (operand_b.op == OpKind::Input &&
                           operand_b.self == TargetArgIndex &&
                           operand_a.op == OpKind::Input &&
                           operand_a.self != TargetArgIndex &&
                           operand_a.self < NumArgs) {
          involves_target = true;
          point_value = fixed_args[operand_a.self];
        }

        // If this comparison involves the target, find Select nodes that use it
        if (involves_target) {
          // Search for Select nodes that use this comparison as their condition
          template for (constexpr auto select_node : nodes) {
            if constexpr (select_node.op == OpKind::Select &&
                          select_node.cond == n.self &&
                          select_node.a < nodes.size() &&
                          select_node.b < nodes.size()) {
              constexpr auto true_branch = nodes[select_node.a];
              constexpr auto false_branch = nodes[select_node.b];

              // Extract amplitude if both branches are constants
              if constexpr (true_branch.op == OpKind::Const &&
                            false_branch.op == OpKind::Const) {
                constexpr double true_val =
                    static_cast<double>([:true_branch.leaf:]);
                constexpr double false_val =
                    static_cast<double>([:false_branch.leaf:]);
                constexpr double amplitude = true_val - false_val;
                collector.add_point_with_amplitude(point_value, amplitude);
              }
            }
          }
        }
      }
    }
  }

  collector.sort_by_points();
  return collector;
}

template <info Fn, std::size_t TargetArgIndex, std::size_t MaxPoints = 16,
          typename... FixedArgs>
consteval DiscontinuityPoints<MaxPoints>
get_discontinuity_points(FixedArgs... fixed_args) {
  constexpr std::size_t NumArgs = TargetArgIndex + 1 + sizeof...(FixedArgs);

  // Create the full input array with fixed values placed after the target
  std::array<double, NumArgs> all_inputs = {};
  all_inputs[TargetArgIndex] = 0.0; // placeholder for target

  // Fill in the fixed values at positions after TargetArgIndex
  // This uses a fold expression to unpack the variadic args
  std::size_t idx = 0;
  ((all_inputs[TargetArgIndex + 1 + idx++] = static_cast<double>(fixed_args)),
   ...);

  // Analyze with fixed arguments
  auto collector =
      analyze_discontinuities_impl<Fn, TargetArgIndex, NumArgs, MaxPoints>(
          all_inputs);
  return collector.get_result();
}

template <info Fn, std::size_t TargetArgIndex, std::size_t MaxPoints = 16,
          typename... FixedArgs>
consteval DiscontinuityPointsWithAmplitudes<MaxPoints>
get_discontinuity_points_and_amplitudes(FixedArgs... fixed_args) {
  constexpr std::size_t NumArgs = TargetArgIndex + 1 + sizeof...(FixedArgs);

  // Create the full input array with fixed values placed after the target
  std::array<double, NumArgs> all_inputs = {};
  all_inputs[TargetArgIndex] = 0.0; // placeholder for target

  // Fill in the fixed values at positions after TargetArgIndex
  // This uses a fold expression to unpack the variadic args
  std::size_t idx = 0;
  ((all_inputs[TargetArgIndex + 1 + idx++] = static_cast<double>(fixed_args)),
   ...);

  // Analyze with fixed arguments to get points and amplitudes
  auto collector =
      analyze_discontinuities_with_amplitudes_impl<Fn, TargetArgIndex, NumArgs,
                                                   MaxPoints>(all_inputs);
  return collector.get_result();
}

} // namespace ad

#endif // DISCONTINUITY_ANALYSIS_HPP
