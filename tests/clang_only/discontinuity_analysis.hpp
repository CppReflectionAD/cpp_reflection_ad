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

        // Case 5: a is target input, b is a binary operation (e.g., strike - 1)
        if constexpr (operand_a.op == OpKind::Input &&
                      operand_a.self == TargetArgIndex &&
                      (operand_b.op == OpKind::Add ||
                       operand_b.op == OpKind::Sub ||
                       operand_b.op == OpKind::Mul ||
                       operand_b.op == OpKind::Div) &&
                      operand_b.a < nodes.size() &&
                      operand_b.b < nodes.size()) {
          // Evaluate the binary operation with fixed arguments
          constexpr auto b_left = nodes[operand_b.a];
          constexpr auto b_right = nodes[operand_b.b];
          double b_left_val = 0.0, b_right_val = 0.0;

          if constexpr (b_left.op == OpKind::Const) {
            b_left_val = static_cast<double>([:b_left.leaf:]);
          } else if constexpr (b_left.op == OpKind::Input &&
                               b_left.self < NumArgs) {
            b_left_val = fixed_args[b_left.self];
          }

          if constexpr (b_right.op == OpKind::Const) {
            b_right_val = static_cast<double>([:b_right.leaf:]);
          } else if constexpr (b_right.op == OpKind::Input &&
                               b_right.self < NumArgs) {
            b_right_val = fixed_args[b_right.self];
          }

          double point_value = 0.0;
          if constexpr (operand_b.op == OpKind::Add) {
            point_value = b_left_val + b_right_val;
          } else if constexpr (operand_b.op == OpKind::Sub) {
            point_value = b_left_val - b_right_val;
          } else if constexpr (operand_b.op == OpKind::Mul) {
            point_value = b_left_val * b_right_val;
          } else if constexpr (operand_b.op == OpKind::Div) {
            point_value =
                (b_right_val != 0.0) ? (b_left_val / b_right_val) : 0.0;
          }
          collector.add_point(point_value);
        }

        // Case 6: b is target input, a is a binary operation (e.g., strike - 1)
        if constexpr (operand_b.op == OpKind::Input &&
                      operand_b.self == TargetArgIndex &&
                      (operand_a.op == OpKind::Add ||
                       operand_a.op == OpKind::Sub ||
                       operand_a.op == OpKind::Mul ||
                       operand_a.op == OpKind::Div) &&
                      operand_a.a < nodes.size() &&
                      operand_a.b < nodes.size()) {
          // Evaluate the binary operation with fixed arguments
          constexpr auto a_left = nodes[operand_a.a];
          constexpr auto a_right = nodes[operand_a.b];
          double a_left_val = 0.0, a_right_val = 0.0;

          if constexpr (a_left.op == OpKind::Const) {
            a_left_val = static_cast<double>([:a_left.leaf:]);
          } else if constexpr (a_left.op == OpKind::Input &&
                               a_left.self < NumArgs) {
            a_left_val = fixed_args[a_left.self];
          }

          if constexpr (a_right.op == OpKind::Const) {
            a_right_val = static_cast<double>([:a_right.leaf:]);
          } else if constexpr (a_right.op == OpKind::Input &&
                               a_right.self < NumArgs) {
            a_right_val = fixed_args[a_right.self];
          }

          double point_value = 0.0;
          if constexpr (operand_a.op == OpKind::Add) {
            point_value = a_left_val + a_right_val;
          } else if constexpr (operand_a.op == OpKind::Sub) {
            point_value = a_left_val - a_right_val;
          } else if constexpr (operand_a.op == OpKind::Mul) {
            point_value = a_left_val * a_right_val;
          } else if constexpr (operand_a.op == OpKind::Div) {
            point_value =
                (a_right_val != 0.0) ? (a_left_val / a_right_val) : 0.0;
          }
          collector.add_point(point_value);
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

        // Case 5: a is target input, b is a binary operation (e.g., strike - 1)
        else if constexpr (operand_a.op == OpKind::Input &&
                           operand_a.self == TargetArgIndex &&
                           (operand_b.op == OpKind::Add ||
                            operand_b.op == OpKind::Sub ||
                            operand_b.op == OpKind::Mul ||
                            operand_b.op == OpKind::Div) &&
                           operand_b.a < nodes.size() &&
                           operand_b.b < nodes.size()) {
          involves_target = true;
          // Evaluate the binary operation with fixed arguments
          constexpr auto b_left = nodes[operand_b.a];
          constexpr auto b_right = nodes[operand_b.b];
          double b_left_val = 0.0, b_right_val = 0.0;

          if constexpr (b_left.op == OpKind::Const) {
            b_left_val = static_cast<double>([:b_left.leaf:]);
          } else if constexpr (b_left.op == OpKind::Input &&
                               b_left.self < NumArgs) {
            b_left_val = fixed_args[b_left.self];
          }

          if constexpr (b_right.op == OpKind::Const) {
            b_right_val = static_cast<double>([:b_right.leaf:]);
          } else if constexpr (b_right.op == OpKind::Input &&
                               b_right.self < NumArgs) {
            b_right_val = fixed_args[b_right.self];
          }

          if constexpr (operand_b.op == OpKind::Add) {
            point_value = b_left_val + b_right_val;
          } else if constexpr (operand_b.op == OpKind::Sub) {
            point_value = b_left_val - b_right_val;
          } else if constexpr (operand_b.op == OpKind::Mul) {
            point_value = b_left_val * b_right_val;
          } else if constexpr (operand_b.op == OpKind::Div) {
            point_value =
                (b_right_val != 0.0) ? (b_left_val / b_right_val) : 0.0;
          }
        }

        // Case 6: b is target input, a is a binary operation (e.g., strike - 1)
        else if constexpr (operand_b.op == OpKind::Input &&
                           operand_b.self == TargetArgIndex &&
                           (operand_a.op == OpKind::Add ||
                            operand_a.op == OpKind::Sub ||
                            operand_a.op == OpKind::Mul ||
                            operand_a.op == OpKind::Div) &&
                           operand_a.a < nodes.size() &&
                           operand_a.b < nodes.size()) {
          involves_target = true;
          // Evaluate the binary operation with fixed arguments
          constexpr auto a_left = nodes[operand_a.a];
          constexpr auto a_right = nodes[operand_a.b];
          double a_left_val = 0.0, a_right_val = 0.0;

          if constexpr (a_left.op == OpKind::Const) {
            a_left_val = static_cast<double>([:a_left.leaf:]);
          } else if constexpr (a_left.op == OpKind::Input &&
                               a_left.self < NumArgs) {
            a_left_val = fixed_args[a_left.self];
          }

          if constexpr (a_right.op == OpKind::Const) {
            a_right_val = static_cast<double>([:a_right.leaf:]);
          } else if constexpr (a_right.op == OpKind::Input &&
                               a_right.self < NumArgs) {
            a_right_val = fixed_args[a_right.self];
          }

          if constexpr (operand_a.op == OpKind::Add) {
            point_value = a_left_val + a_right_val;
          } else if constexpr (operand_a.op == OpKind::Sub) {
            point_value = a_left_val - a_right_val;
          } else if constexpr (operand_a.op == OpKind::Mul) {
            point_value = a_left_val * a_right_val;
          } else if constexpr (operand_a.op == OpKind::Div) {
            point_value =
                (a_right_val != 0.0) ? (a_left_val / a_right_val) : 0.0;
          }
        }

        // If this comparison involves the target, find Select nodes that use it
        if (involves_target) {
          // Search for Select nodes that use this comparison as their condition
          template for (constexpr auto select_node : nodes) {
            if constexpr (select_node.op == OpKind::Select &&
                          select_node.cond == n.self &&
                          select_node.a < nodes.size() &&
                          select_node.b < nodes.size()) {
              // Evaluate true branch (select_node.a) at point_value
              double true_val = 0.0;
              constexpr auto true_node = nodes[select_node.a];

              if constexpr (true_node.op == OpKind::Const) {
                true_val = static_cast<double>([:true_node.leaf:]);
              } else if constexpr (true_node.op == OpKind::Neg &&
                                   true_node.a < nodes.size()) {
                // Handle unary negation: -expr
                constexpr auto neg_operand = nodes[true_node.a];
                if constexpr (neg_operand.op == OpKind::Const) {
                  true_val = -static_cast<double>([:neg_operand.leaf:]);
                }
              } else if constexpr (true_node.op == OpKind::Input) {
                true_val =
                    (true_node.self == TargetArgIndex)
                        ? point_value
                        : (true_node.self < NumArgs ? fixed_args[true_node.self]
                                                    : 0.0);
              } else if constexpr (true_node.op == OpKind::Add ||
                                   true_node.op == OpKind::Sub ||
                                   true_node.op == OpKind::Mul ||
                                   true_node.op == OpKind::Div) {
                // Binary operation - evaluate its operands
                if constexpr (true_node.a < nodes.size() &&
                              true_node.b < nodes.size()) {
                  constexpr auto ta = nodes[true_node.a];
                  constexpr auto tb = nodes[true_node.b];

                  double ta_val = 0.0, tb_val = 0.0;
                  // Evaluate first operand
                  if constexpr (ta.op == OpKind::Const) {
                    ta_val = static_cast<double>([:ta.leaf:]);
                  } else if constexpr (ta.op == OpKind::Input) {
                    ta_val =
                        (ta.self == TargetArgIndex)
                            ? point_value
                            : (ta.self < NumArgs ? fixed_args[ta.self] : 0.0);
                  } else if constexpr (ta.op == OpKind::Add ||
                                       ta.op == OpKind::Sub ||
                                       ta.op == OpKind::Mul ||
                                       ta.op == OpKind::Div) {
                    // Nested binary op on left
                    if constexpr (ta.a < nodes.size() && ta.b < nodes.size()) {
                      constexpr auto taa = nodes[ta.a];
                      constexpr auto tab = nodes[ta.b];
                      double taa_val = 0.0;
                      if constexpr (taa.op == OpKind::Const) {
                        taa_val = static_cast<double>([:taa.leaf:]);
                      } else if constexpr (taa.op == OpKind::Input) {
                        taa_val =
                            (taa.self == TargetArgIndex)
                                ? point_value
                                : (taa.self < NumArgs ? fixed_args[taa.self]
                                                      : 0.0);
                      }
                      double tab_val = 0.0;
                      if constexpr (tab.op == OpKind::Const) {
                        tab_val = static_cast<double>([:tab.leaf:]);
                      } else if constexpr (tab.op == OpKind::Input) {
                        tab_val =
                            (tab.self == TargetArgIndex)
                                ? point_value
                                : (tab.self < NumArgs ? fixed_args[tab.self]
                                                      : 0.0);
                      }
                      if constexpr (ta.op == OpKind::Add) {
                        ta_val = taa_val + tab_val;
                      } else if constexpr (ta.op == OpKind::Sub) {
                        ta_val = taa_val - tab_val;
                      } else if constexpr (ta.op == OpKind::Mul) {
                        ta_val = taa_val * tab_val;
                      } else if constexpr (ta.op == OpKind::Div) {
                        ta_val = (tab_val != 0.0) ? (taa_val / tab_val) : 0.0;
                      }
                    }
                  }

                  // Evaluate second operand
                  if constexpr (tb.op == OpKind::Const) {
                    tb_val = static_cast<double>([:tb.leaf:]);
                  } else if constexpr (tb.op == OpKind::Input) {
                    tb_val =
                        (tb.self == TargetArgIndex)
                            ? point_value
                            : (tb.self < NumArgs ? fixed_args[tb.self] : 0.0);
                  } else if constexpr (tb.op == OpKind::Add ||
                                       tb.op == OpKind::Sub ||
                                       tb.op == OpKind::Mul ||
                                       tb.op == OpKind::Div) {
                    // Nested binary op on right
                    if constexpr (tb.a < nodes.size() && tb.b < nodes.size()) {
                      constexpr auto tba = nodes[tb.a];
                      constexpr auto tbb = nodes[tb.b];
                      double tba_val = 0.0;
                      if constexpr (tba.op == OpKind::Const) {
                        tba_val = static_cast<double>([:tba.leaf:]);
                      } else if constexpr (tba.op == OpKind::Input) {
                        tba_val =
                            (tba.self == TargetArgIndex)
                                ? point_value
                                : (tba.self < NumArgs ? fixed_args[tba.self]
                                                      : 0.0);
                      }
                      double tbb_val = 0.0;
                      if constexpr (tbb.op == OpKind::Const) {
                        tbb_val = static_cast<double>([:tbb.leaf:]);
                      } else if constexpr (tbb.op == OpKind::Input) {
                        tbb_val =
                            (tbb.self == TargetArgIndex)
                                ? point_value
                                : (tbb.self < NumArgs ? fixed_args[tbb.self]
                                                      : 0.0);
                      }
                      if constexpr (tb.op == OpKind::Add) {
                        tb_val = tba_val + tbb_val;
                      } else if constexpr (tb.op == OpKind::Sub) {
                        tb_val = tba_val - tbb_val;
                      } else if constexpr (tb.op == OpKind::Mul) {
                        tb_val = tba_val * tbb_val;
                      } else if constexpr (tb.op == OpKind::Div) {
                        tb_val = (tbb_val != 0.0) ? (tba_val / tbb_val) : 0.0;
                      }
                    }
                  }

                  if constexpr (true_node.op == OpKind::Add) {
                    true_val = ta_val + tb_val;
                  } else if constexpr (true_node.op == OpKind::Sub) {
                    true_val = ta_val - tb_val;
                  } else if constexpr (true_node.op == OpKind::Mul) {
                    true_val = ta_val * tb_val;
                  } else if constexpr (true_node.op == OpKind::Div) {
                    true_val = (tb_val != 0.0) ? (ta_val / tb_val) : 0.0;
                  }
                }
              }

              // Evaluate false branch (select_node.b) at point_value
              double false_val = 0.0;
              constexpr auto false_node = nodes[select_node.b];

              if constexpr (false_node.op == OpKind::Const) {
                false_val = static_cast<double>([:false_node.leaf:]);
              } else if constexpr (false_node.op == OpKind::Neg &&
                                   false_node.a < nodes.size()) {
                // Handle unary negation: -expr
                constexpr auto neg_operand = nodes[false_node.a];
                if constexpr (neg_operand.op == OpKind::Const) {
                  false_val = -static_cast<double>([:neg_operand.leaf:]);
                }
              } else if constexpr (false_node.op == OpKind::Input) {
                false_val = (false_node.self == TargetArgIndex)
                                ? point_value
                                : (false_node.self < NumArgs
                                       ? fixed_args[false_node.self]
                                       : 0.0);
              } else if constexpr (false_node.op == OpKind::Add ||
                                   false_node.op == OpKind::Sub ||
                                   false_node.op == OpKind::Mul ||
                                   false_node.op == OpKind::Div) {
                // Binary operation - evaluate its operands
                if constexpr (false_node.a < nodes.size() &&
                              false_node.b < nodes.size()) {
                  constexpr auto fa = nodes[false_node.a];
                  constexpr auto fb = nodes[false_node.b];

                  double fa_val = 0.0, fb_val = 0.0;
                  // Evaluate first operand
                  if constexpr (fa.op == OpKind::Const) {
                    fa_val = static_cast<double>([:fa.leaf:]);
                  } else if constexpr (fa.op == OpKind::Input) {
                    fa_val =
                        (fa.self == TargetArgIndex)
                            ? point_value
                            : (fa.self < NumArgs ? fixed_args[fa.self] : 0.0);
                  } else if constexpr (fa.op == OpKind::Add ||
                                       fa.op == OpKind::Sub ||
                                       fa.op == OpKind::Mul ||
                                       fa.op == OpKind::Div) {
                    // Nested binary op on left
                    if constexpr (fa.a < nodes.size() && fa.b < nodes.size()) {
                      constexpr auto faa = nodes[fa.a];
                      constexpr auto fab = nodes[fa.b];
                      double faa_val = 0.0;
                      if constexpr (faa.op == OpKind::Const) {
                        faa_val = static_cast<double>([:faa.leaf:]);
                      } else if constexpr (faa.op == OpKind::Input) {
                        faa_val =
                            (faa.self == TargetArgIndex)
                                ? point_value
                                : (faa.self < NumArgs ? fixed_args[faa.self]
                                                      : 0.0);
                      }
                      double fab_val = 0.0;
                      if constexpr (fab.op == OpKind::Const) {
                        fab_val = static_cast<double>([:fab.leaf:]);
                      } else if constexpr (fab.op == OpKind::Input) {
                        fab_val =
                            (fab.self == TargetArgIndex)
                                ? point_value
                                : (fab.self < NumArgs ? fixed_args[fab.self]
                                                      : 0.0);
                      }
                      if constexpr (fa.op == OpKind::Add) {
                        fa_val = faa_val + fab_val;
                      } else if constexpr (fa.op == OpKind::Sub) {
                        fa_val = faa_val - fab_val;
                      } else if constexpr (fa.op == OpKind::Mul) {
                        fa_val = faa_val * fab_val;
                      } else if constexpr (fa.op == OpKind::Div) {
                        fa_val = (fab_val != 0.0) ? (faa_val / fab_val) : 0.0;
                      }
                    }
                  }

                  // Evaluate second operand
                  if constexpr (fb.op == OpKind::Const) {
                    fb_val = static_cast<double>([:fb.leaf:]);
                  } else if constexpr (fb.op == OpKind::Input) {
                    fb_val =
                        (fb.self == TargetArgIndex)
                            ? point_value
                            : (fb.self < NumArgs ? fixed_args[fb.self] : 0.0);
                  } else if constexpr (fb.op == OpKind::Add ||
                                       fb.op == OpKind::Sub ||
                                       fb.op == OpKind::Mul ||
                                       fb.op == OpKind::Div) {
                    // Nested binary op on right
                    if constexpr (fb.a < nodes.size() && fb.b < nodes.size()) {
                      constexpr auto fba = nodes[fb.a];
                      constexpr auto fbb = nodes[fb.b];
                      double fba_val = 0.0;
                      if constexpr (fba.op == OpKind::Const) {
                        fba_val = static_cast<double>([:fba.leaf:]);
                      } else if constexpr (fba.op == OpKind::Input) {
                        fba_val =
                            (fba.self == TargetArgIndex)
                                ? point_value
                                : (fba.self < NumArgs ? fixed_args[fba.self]
                                                      : 0.0);
                      }
                      double fbb_val = 0.0;
                      if constexpr (fbb.op == OpKind::Const) {
                        fbb_val = static_cast<double>([:fbb.leaf:]);
                      } else if constexpr (fbb.op == OpKind::Input) {
                        fbb_val =
                            (fbb.self == TargetArgIndex)
                                ? point_value
                                : (fbb.self < NumArgs ? fixed_args[fbb.self]
                                                      : 0.0);
                      }
                      if constexpr (fb.op == OpKind::Add) {
                        fb_val = fba_val + fbb_val;
                      } else if constexpr (fb.op == OpKind::Sub) {
                        fb_val = fba_val - fbb_val;
                      } else if constexpr (fb.op == OpKind::Mul) {
                        fb_val = fba_val * fbb_val;
                      } else if constexpr (fb.op == OpKind::Div) {
                        fb_val = (fbb_val != 0.0) ? (fba_val / fbb_val) : 0.0;
                      }
                    }
                  }

                  if constexpr (false_node.op == OpKind::Add) {
                    false_val = fa_val + fb_val;
                  } else if constexpr (false_node.op == OpKind::Sub) {
                    false_val = fa_val - fb_val;
                  } else if constexpr (false_node.op == OpKind::Mul) {
                    false_val = fa_val * fb_val;
                  } else if constexpr (false_node.op == OpKind::Div) {
                    false_val = (fb_val != 0.0) ? (fa_val / fb_val) : 0.0;
                  }
                }
              }

              double amplitude = true_val - false_val;

              // Check if this Select node is used in any binary operations that
              // would affect its amplitude
              double amplitude_scale = 1.0;
              template for (constexpr auto parent_node : nodes) {
                if constexpr ((parent_node.op == OpKind::Add ||
                               parent_node.op == OpKind::Sub ||
                               parent_node.op == OpKind::Mul ||
                               parent_node.op == OpKind::Div) &&
                              (parent_node.a == select_node.self ||
                               parent_node.b == select_node.self)) {
                  // This Select is used as an operand in a binary operation
                  if constexpr (parent_node.op == OpKind::Sub &&
                                parent_node.b == select_node.self) {
                    // This Select is being subtracted (it's the right operand
                    // of Sub)
                    amplitude_scale = -1.0;
                  }
                  // For Add/Mul/Div or left operand of Sub, amplitude_scale
                  // stays 1.0
                }
              }

              amplitude = amplitude * amplitude_scale;
              collector.add_point_with_amplitude(point_value, amplitude);
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

// Runtime version: accepts dynamic (non-constexpr) fixed arguments
// Performs compile-time DAG analysis but uses runtime parameter values
template <info Fn, std::size_t TargetArgIndex, std::size_t MaxPoints = 16,
          typename... FixedArgs>
inline DiscontinuityPointsWithAmplitudes<MaxPoints>
get_discontinuity_points_and_amplitudes_rt(FixedArgs... fixed_args) {
  constexpr std::size_t NumArgs = TargetArgIndex + 1 + sizeof...(FixedArgs);

  // Create the full input array with fixed values placed after the target
  std::array<double, NumArgs> all_inputs = {};
  all_inputs[TargetArgIndex] = 0.0; // placeholder for target

  // Fill in the fixed values at positions after TargetArgIndex
  std::size_t idx = 0;
  ((all_inputs[TargetArgIndex + 1 + idx++] = static_cast<double>(fixed_args)),
   ...);

  // Perform compile-time analysis on the DAG structure
  // Extract discontinuity points and amplitudes
  DiscontinuityPointsWithAmplitudes<MaxPoints> result{};

  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());

  // Template for loop processes DAG structure at compile-time
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
          // Use the runtime value of operand_b
          point_value = all_inputs[operand_b.self];
        }

        // Case 4: b is target input, a is another input (non-target)
        else if constexpr (operand_b.op == OpKind::Input &&
                           operand_b.self == TargetArgIndex &&
                           operand_a.op == OpKind::Input &&
                           operand_a.self != TargetArgIndex &&
                           operand_a.self < NumArgs) {
          involves_target = true;
          // Use the runtime value of operand_a
          point_value = all_inputs[operand_a.self];
        }

        // Case 5: a is target input, b is a binary operation (e.g., strike - 1)
        else if constexpr (operand_a.op == OpKind::Input &&
                           operand_a.self == TargetArgIndex &&
                           (operand_b.op == OpKind::Add ||
                            operand_b.op == OpKind::Sub ||
                            operand_b.op == OpKind::Mul ||
                            operand_b.op == OpKind::Div) &&
                           operand_b.a < nodes.size() &&
                           operand_b.b < nodes.size()) {
          involves_target = true;
          // Evaluate the binary operation with fixed arguments
          constexpr auto b_left = nodes[operand_b.a];
          constexpr auto b_right = nodes[operand_b.b];
          double b_left_val = 0.0, b_right_val = 0.0;

          if constexpr (b_left.op == OpKind::Const) {
            b_left_val = static_cast<double>([:b_left.leaf:]);
          } else if constexpr (b_left.op == OpKind::Input &&
                               b_left.self < NumArgs) {
            b_left_val = all_inputs[b_left.self];
          }

          if constexpr (b_right.op == OpKind::Const) {
            b_right_val = static_cast<double>([:b_right.leaf:]);
          } else if constexpr (b_right.op == OpKind::Input &&
                               b_right.self < NumArgs) {
            b_right_val = all_inputs[b_right.self];
          }

          if constexpr (operand_b.op == OpKind::Add) {
            point_value = b_left_val + b_right_val;
          } else if constexpr (operand_b.op == OpKind::Sub) {
            point_value = b_left_val - b_right_val;
          } else if constexpr (operand_b.op == OpKind::Mul) {
            point_value = b_left_val * b_right_val;
          } else if constexpr (operand_b.op == OpKind::Div) {
            point_value =
                (b_right_val != 0.0) ? (b_left_val / b_right_val) : 0.0;
          }
        }

        // Case 6: b is target input, a is a binary operation (e.g., strike - 1)
        else if constexpr (operand_b.op == OpKind::Input &&
                           operand_b.self == TargetArgIndex &&
                           (operand_a.op == OpKind::Add ||
                            operand_a.op == OpKind::Sub ||
                            operand_a.op == OpKind::Mul ||
                            operand_a.op == OpKind::Div) &&
                           operand_a.a < nodes.size() &&
                           operand_a.b < nodes.size()) {
          involves_target = true;
          // Evaluate the binary operation with fixed arguments
          constexpr auto a_left = nodes[operand_a.a];
          constexpr auto a_right = nodes[operand_a.b];
          double a_left_val = 0.0, a_right_val = 0.0;

          if constexpr (a_left.op == OpKind::Const) {
            a_left_val = static_cast<double>([:a_left.leaf:]);
          } else if constexpr (a_left.op == OpKind::Input &&
                               a_left.self < NumArgs) {
            a_left_val = all_inputs[a_left.self];
          }

          if constexpr (a_right.op == OpKind::Const) {
            a_right_val = static_cast<double>([:a_right.leaf:]);
          } else if constexpr (a_right.op == OpKind::Input &&
                               a_right.self < NumArgs) {
            a_right_val = all_inputs[a_right.self];
          }

          if constexpr (operand_a.op == OpKind::Add) {
            point_value = a_left_val + a_right_val;
          } else if constexpr (operand_a.op == OpKind::Sub) {
            point_value = a_left_val - a_right_val;
          } else if constexpr (operand_a.op == OpKind::Mul) {
            point_value = a_left_val * a_right_val;
          } else if constexpr (operand_a.op == OpKind::Div) {
            point_value =
                (a_right_val != 0.0) ? (a_left_val / a_right_val) : 0.0;
          }
        }

        // If this comparison involves the target, find Select nodes that use it
        if (involves_target && result.count < MaxPoints) {
          // Search for Select nodes that use this comparison as their condition
          template for (constexpr auto select_node : nodes) {
            if constexpr (select_node.op == OpKind::Select &&
                          select_node.cond == n.self &&
                          select_node.a < nodes.size() &&
                          select_node.b < nodes.size()) {
              // Evaluate true branch (select_node.a) at point_value
              double true_val = 0.0;
              constexpr auto true_node = nodes[select_node.a];

              if constexpr (true_node.op == OpKind::Const) {
                true_val = static_cast<double>([:true_node.leaf:]);
              } else if constexpr (true_node.op == OpKind::Neg &&
                                   true_node.a < nodes.size()) {
                // Handle unary negation: -expr
                constexpr auto neg_operand = nodes[true_node.a];
                if constexpr (neg_operand.op == OpKind::Const) {
                  true_val = -static_cast<double>([:neg_operand.leaf:]);
                }
              } else if constexpr (true_node.op == OpKind::Input) {
                true_val =
                    (true_node.self == TargetArgIndex)
                        ? point_value
                        : (true_node.self < NumArgs ? all_inputs[true_node.self]
                                                    : 0.0);
              } else if constexpr (true_node.op == OpKind::Add ||
                                   true_node.op == OpKind::Sub ||
                                   true_node.op == OpKind::Mul ||
                                   true_node.op == OpKind::Div) {
                // Binary operation - evaluate its operands (same logic as
                // constexpr version)
                if constexpr (true_node.a < nodes.size() &&
                              true_node.b < nodes.size()) {
                  constexpr auto ta = nodes[true_node.a];
                  constexpr auto tb = nodes[true_node.b];

                  double ta_val = 0.0, tb_val = 0.0;
                  // Evaluate first operand
                  if constexpr (ta.op == OpKind::Const) {
                    ta_val = static_cast<double>([:ta.leaf:]);
                  } else if constexpr (ta.op == OpKind::Input) {
                    ta_val =
                        (ta.self == TargetArgIndex)
                            ? point_value
                            : (ta.self < NumArgs ? all_inputs[ta.self] : 0.0);
                  } else if constexpr (ta.op == OpKind::Add ||
                                       ta.op == OpKind::Sub ||
                                       ta.op == OpKind::Mul ||
                                       ta.op == OpKind::Div) {
                    // Nested binary op on left
                    if constexpr (ta.a < nodes.size() && ta.b < nodes.size()) {
                      constexpr auto taa = nodes[ta.a];
                      constexpr auto tab = nodes[ta.b];
                      double taa_val = 0.0;
                      if constexpr (taa.op == OpKind::Const) {
                        taa_val = static_cast<double>([:taa.leaf:]);
                      } else if constexpr (taa.op == OpKind::Input) {
                        taa_val =
                            (taa.self == TargetArgIndex)
                                ? point_value
                                : (taa.self < NumArgs ? all_inputs[taa.self]
                                                      : 0.0);
                      }
                      double tab_val = 0.0;
                      if constexpr (tab.op == OpKind::Const) {
                        tab_val = static_cast<double>([:tab.leaf:]);
                      } else if constexpr (tab.op == OpKind::Input) {
                        tab_val =
                            (tab.self == TargetArgIndex)
                                ? point_value
                                : (tab.self < NumArgs ? all_inputs[tab.self]
                                                      : 0.0);
                      }
                      if constexpr (ta.op == OpKind::Add) {
                        ta_val = taa_val + tab_val;
                      } else if constexpr (ta.op == OpKind::Sub) {
                        ta_val = taa_val - tab_val;
                      } else if constexpr (ta.op == OpKind::Mul) {
                        ta_val = taa_val * tab_val;
                      } else if constexpr (ta.op == OpKind::Div) {
                        ta_val = (tab_val != 0.0) ? (taa_val / tab_val) : 0.0;
                      }
                    }
                  }

                  // Evaluate second operand
                  if constexpr (tb.op == OpKind::Const) {
                    tb_val = static_cast<double>([:tb.leaf:]);
                  } else if constexpr (tb.op == OpKind::Input) {
                    tb_val =
                        (tb.self == TargetArgIndex)
                            ? point_value
                            : (tb.self < NumArgs ? all_inputs[tb.self] : 0.0);
                  } else if constexpr (tb.op == OpKind::Add ||
                                       tb.op == OpKind::Sub ||
                                       tb.op == OpKind::Mul ||
                                       tb.op == OpKind::Div) {
                    // Nested binary op on right
                    if constexpr (tb.a < nodes.size() && tb.b < nodes.size()) {
                      constexpr auto tba = nodes[tb.a];
                      constexpr auto tbb = nodes[tb.b];
                      double tba_val = 0.0;
                      if constexpr (tba.op == OpKind::Const) {
                        tba_val = static_cast<double>([:tba.leaf:]);
                      } else if constexpr (tba.op == OpKind::Input) {
                        tba_val =
                            (tba.self == TargetArgIndex)
                                ? point_value
                                : (tba.self < NumArgs ? all_inputs[tba.self]
                                                      : 0.0);
                      }
                      double tbb_val = 0.0;
                      if constexpr (tbb.op == OpKind::Const) {
                        tbb_val = static_cast<double>([:tbb.leaf:]);
                      } else if constexpr (tbb.op == OpKind::Input) {
                        tbb_val =
                            (tbb.self == TargetArgIndex)
                                ? point_value
                                : (tbb.self < NumArgs ? all_inputs[tbb.self]
                                                      : 0.0);
                      }
                      if constexpr (tb.op == OpKind::Add) {
                        tb_val = tba_val + tbb_val;
                      } else if constexpr (tb.op == OpKind::Sub) {
                        tb_val = tba_val - tbb_val;
                      } else if constexpr (tb.op == OpKind::Mul) {
                        tb_val = tba_val * tbb_val;
                      } else if constexpr (tb.op == OpKind::Div) {
                        tb_val = (tbb_val != 0.0) ? (tba_val / tbb_val) : 0.0;
                      }
                    }
                  }

                  if constexpr (true_node.op == OpKind::Add) {
                    true_val = ta_val + tb_val;
                  } else if constexpr (true_node.op == OpKind::Sub) {
                    true_val = ta_val - tb_val;
                  } else if constexpr (true_node.op == OpKind::Mul) {
                    true_val = ta_val * tb_val;
                  } else if constexpr (true_node.op == OpKind::Div) {
                    true_val = (tb_val != 0.0) ? (ta_val / tb_val) : 0.0;
                  }
                }
              }

              // Evaluate false branch (select_node.b) at point_value
              double false_val = 0.0;
              constexpr auto false_node = nodes[select_node.b];

              if constexpr (false_node.op == OpKind::Const) {
                false_val = static_cast<double>([:false_node.leaf:]);
              } else if constexpr (false_node.op == OpKind::Neg &&
                                   false_node.a < nodes.size()) {
                // Handle unary negation: -expr
                constexpr auto neg_operand = nodes[false_node.a];
                if constexpr (neg_operand.op == OpKind::Const) {
                  false_val = -static_cast<double>([:neg_operand.leaf:]);
                }
              } else if constexpr (false_node.op == OpKind::Input) {
                false_val = (false_node.self == TargetArgIndex)
                                ? point_value
                                : (false_node.self < NumArgs
                                       ? all_inputs[false_node.self]
                                       : 0.0);
              } else if constexpr (false_node.op == OpKind::Add ||
                                   false_node.op == OpKind::Sub ||
                                   false_node.op == OpKind::Mul ||
                                   false_node.op == OpKind::Div) {
                // Binary operation - evaluate its operands
                if constexpr (false_node.a < nodes.size() &&
                              false_node.b < nodes.size()) {
                  constexpr auto fa = nodes[false_node.a];
                  constexpr auto fb = nodes[false_node.b];

                  double fa_val = 0.0, fb_val = 0.0;
                  // Evaluate first operand
                  if constexpr (fa.op == OpKind::Const) {
                    fa_val = static_cast<double>([:fa.leaf:]);
                  } else if constexpr (fa.op == OpKind::Input) {
                    fa_val =
                        (fa.self == TargetArgIndex)
                            ? point_value
                            : (fa.self < NumArgs ? all_inputs[fa.self] : 0.0);
                  } else if constexpr (fa.op == OpKind::Add ||
                                       fa.op == OpKind::Sub ||
                                       fa.op == OpKind::Mul ||
                                       fa.op == OpKind::Div) {
                    // Nested binary op on left
                    if constexpr (fa.a < nodes.size() && fa.b < nodes.size()) {
                      constexpr auto faa = nodes[fa.a];
                      constexpr auto fab = nodes[fa.b];
                      double faa_val = 0.0;
                      if constexpr (faa.op == OpKind::Const) {
                        faa_val = static_cast<double>([:faa.leaf:]);
                      } else if constexpr (faa.op == OpKind::Input) {
                        faa_val =
                            (faa.self == TargetArgIndex)
                                ? point_value
                                : (faa.self < NumArgs ? all_inputs[faa.self]
                                                      : 0.0);
                      }
                      double fab_val = 0.0;
                      if constexpr (fab.op == OpKind::Const) {
                        fab_val = static_cast<double>([:fab.leaf:]);
                      } else if constexpr (fab.op == OpKind::Input) {
                        fab_val =
                            (fab.self == TargetArgIndex)
                                ? point_value
                                : (fab.self < NumArgs ? all_inputs[fab.self]
                                                      : 0.0);
                      }
                      if constexpr (fa.op == OpKind::Add) {
                        fa_val = faa_val + fab_val;
                      } else if constexpr (fa.op == OpKind::Sub) {
                        fa_val = faa_val - fab_val;
                      } else if constexpr (fa.op == OpKind::Mul) {
                        fa_val = faa_val * fab_val;
                      } else if constexpr (fa.op == OpKind::Div) {
                        fa_val = (fab_val != 0.0) ? (faa_val / fab_val) : 0.0;
                      }
                    }
                  }

                  // Evaluate second operand
                  if constexpr (fb.op == OpKind::Const) {
                    fb_val = static_cast<double>([:fb.leaf:]);
                  } else if constexpr (fb.op == OpKind::Input) {
                    fb_val =
                        (fb.self == TargetArgIndex)
                            ? point_value
                            : (fb.self < NumArgs ? all_inputs[fb.self] : 0.0);
                  } else if constexpr (fb.op == OpKind::Add ||
                                       fb.op == OpKind::Sub ||
                                       fb.op == OpKind::Mul ||
                                       fb.op == OpKind::Div) {
                    // Nested binary op on right
                    if constexpr (fb.a < nodes.size() && fb.b < nodes.size()) {
                      constexpr auto fba = nodes[fb.a];
                      constexpr auto fbb = nodes[fb.b];
                      double fba_val = 0.0;
                      if constexpr (fba.op == OpKind::Const) {
                        fba_val = static_cast<double>([:fba.leaf:]);
                      } else if constexpr (fba.op == OpKind::Input) {
                        fba_val =
                            (fba.self == TargetArgIndex)
                                ? point_value
                                : (fba.self < NumArgs ? all_inputs[fba.self]
                                                      : 0.0);
                      }
                      double fbb_val = 0.0;
                      if constexpr (fbb.op == OpKind::Const) {
                        fbb_val = static_cast<double>([:fbb.leaf:]);
                      } else if constexpr (fbb.op == OpKind::Input) {
                        fbb_val =
                            (fbb.self == TargetArgIndex)
                                ? point_value
                                : (fbb.self < NumArgs ? all_inputs[fbb.self]
                                                      : 0.0);
                      }
                      if constexpr (fb.op == OpKind::Add) {
                        fb_val = fba_val + fbb_val;
                      } else if constexpr (fb.op == OpKind::Sub) {
                        fb_val = fba_val - fbb_val;
                      } else if constexpr (fb.op == OpKind::Mul) {
                        fb_val = fba_val * fbb_val;
                      } else if constexpr (fb.op == OpKind::Div) {
                        fb_val = (fbb_val != 0.0) ? (fba_val / fbb_val) : 0.0;
                      }
                    }
                  }

                  if constexpr (false_node.op == OpKind::Add) {
                    false_val = fa_val + fb_val;
                  } else if constexpr (false_node.op == OpKind::Sub) {
                    false_val = fa_val - fb_val;
                  } else if constexpr (false_node.op == OpKind::Mul) {
                    false_val = fa_val * fb_val;
                  } else if constexpr (false_node.op == OpKind::Div) {
                    false_val = (fb_val != 0.0) ? (fa_val / fb_val) : 0.0;
                  }
                }
              }

              double amplitude = true_val - false_val;

              // Check if this Select node is used in any binary operations that
              // would affect its amplitude
              double amplitude_scale = 1.0;
              template for (constexpr auto parent_node : nodes) {
                if constexpr ((parent_node.op == OpKind::Add ||
                               parent_node.op == OpKind::Sub ||
                               parent_node.op == OpKind::Mul ||
                               parent_node.op == OpKind::Div) &&
                              (parent_node.a == select_node.self ||
                               parent_node.b == select_node.self)) {
                  // This Select is used as an operand in a binary operation
                  if constexpr (parent_node.op == OpKind::Sub &&
                                parent_node.b == select_node.self) {
                    // This Select is being subtracted (it's the right operand
                    // of Sub)
                    amplitude_scale = -1.0;
                  }
                  // For Add/Mul/Div or left operand of Sub, amplitude_scale
                  // stays 1.0
                }
              }

              amplitude = amplitude * amplitude_scale;

              // Check for duplicates before adding
              bool found = false;
              for (std::size_t j = 0; j < result.count; ++j) {
                if (result.point(j) == point_value) {
                  found = true;
                  break;
                }
              }
              if (!found) {
                result.data[result.count * 2] = point_value;
                result.data[result.count * 2 + 1] = amplitude;
                result.count++;
              }
            }
          }
        }
      }
    }
  }

  // Sort by points
  for (std::size_t i = 0; i < result.count; ++i) {
    for (std::size_t j = i + 1; j < result.count; ++j) {
      if (result.point(j) < result.point(i)) {
        // Swap points
        double tmp_p = result.data[i * 2];
        result.data[i * 2] = result.data[j * 2];
        result.data[j * 2] = tmp_p;
        // Swap amplitudes
        double tmp_a = result.data[i * 2 + 1];
        result.data[i * 2 + 1] = result.data[j * 2 + 1];
        result.data[j * 2 + 1] = tmp_a;
      }
    }
  }

  return result;
}

} // namespace ad

#endif // DISCONTINUITY_ANALYSIS_HPP
