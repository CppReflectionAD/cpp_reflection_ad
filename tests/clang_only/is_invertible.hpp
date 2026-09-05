#ifndef IS_INVERTIBLE_HPP
#define IS_INVERTIBLE_HPP

// is_invertible.hpp - conservative compile-time invertibility checker.
//
// This walks the SSA/DAG from autograd.h and proves injectivity only for a
// restricted scalar fragment on its largest natural domain:
// - affine maps a*x + b with a != 0
// - compositions with globally injective unary ops (exp, log, sqrt, erfc, neg)
// - add/sub by constants, mul/div by nonzero constants
//
// The checker is intentionally sound-but-incomplete: it returns false when it
// cannot prove invertibility from structure alone.

#include "../autograd.h" // for ad::Node, ad::OpKind, ad::build_nodes<>

#include <array>
#include <cstddef>
#include <type_traits>

namespace ad {

namespace detail_inv {

struct NodeInfo {
  bool invertible = false;
  bool depends_input = false;

  bool is_constant = false;
  double constant_value = 0.0;

  bool is_affine = false;
  double slope = 0.0;
  double intercept = 0.0;
};

consteval bool is_nonzero(double x) { return x != 0.0; }

consteval NodeInfo make_const(double v) {
  return NodeInfo{
      .invertible = false,
      .depends_input = false,
      .is_constant = true,
      .constant_value = v,
      .is_affine = true,
      .slope = 0.0,
      .intercept = v,
  };
}

consteval NodeInfo make_input() {
  return NodeInfo{
      .invertible = true,
      .depends_input = true,
      .is_constant = false,
      .constant_value = 0.0,
      .is_affine = true,
      .slope = 1.0,
      .intercept = 0.0,
  };
}

consteval NodeInfo combine_add(const NodeInfo &a, const NodeInfo &b,
                               bool is_sub) {
  NodeInfo out{};
  out.depends_input = a.depends_input || b.depends_input;
  out.is_constant = a.is_constant && b.is_constant;
  if (out.is_constant)
    out.constant_value = is_sub ? (a.constant_value - b.constant_value)
                                : (a.constant_value + b.constant_value);

  if (a.is_affine && b.is_affine) {
    out.is_affine = true;
    out.slope = is_sub ? (a.slope - b.slope) : (a.slope + b.slope);
    out.intercept =
        is_sub ? (a.intercept - b.intercept) : (a.intercept + b.intercept);
    out.invertible = out.depends_input && is_nonzero(out.slope);
    return out;
  }

  out.is_affine = false;
  if (a.is_constant && !b.is_constant) {
    out.invertible = b.invertible;
  } else if (!a.is_constant && b.is_constant) {
    out.invertible = a.invertible;
  } else {
    out.invertible = false;
  }
  return out;
}

consteval NodeInfo combine_mul(const NodeInfo &a, const NodeInfo &b) {
  NodeInfo out{};
  out.depends_input = a.depends_input || b.depends_input;
  out.is_constant = a.is_constant && b.is_constant;
  if (out.is_constant)
    out.constant_value = a.constant_value * b.constant_value;

  // Affine only when exactly one side is constant.
  if (a.is_constant && b.is_affine) {
    out.is_affine = true;
    out.slope = a.constant_value * b.slope;
    out.intercept = a.constant_value * b.intercept;
  } else if (b.is_constant && a.is_affine) {
    out.is_affine = true;
    out.slope = b.constant_value * a.slope;
    out.intercept = b.constant_value * a.intercept;
  } else {
    out.is_affine = false;
  }

  if (a.is_constant && !b.is_constant) {
    out.invertible = is_nonzero(a.constant_value) && b.invertible;
  } else if (!a.is_constant && b.is_constant) {
    out.invertible = is_nonzero(b.constant_value) && a.invertible;
  } else {
    // Includes x*x (non-injective on largest domain).
    out.invertible = false;
  }

  return out;
}

consteval NodeInfo combine_div(const NodeInfo &a, const NodeInfo &b) {
  NodeInfo out{};
  out.depends_input = a.depends_input || b.depends_input;
  out.is_constant = a.is_constant && b.is_constant;
  if (out.is_constant)
    out.constant_value = a.constant_value / b.constant_value;

  if (b.is_constant && is_nonzero(b.constant_value) && a.is_affine) {
    out.is_affine = true;
    out.slope = a.slope / b.constant_value;
    out.intercept = a.intercept / b.constant_value;
  } else {
    out.is_affine = false;
  }

  // Conservative rules:
  // 1) f/c is invertible when c != 0 and f is invertible.
  // 2) c/f is invertible when c != 0 and f is invertible, interpreted on the
  //    largest natural domain where f(x) != 0.
  if (b.is_constant && is_nonzero(b.constant_value)) {
    out.invertible = a.invertible;
  } else if (a.is_constant && is_nonzero(a.constant_value)) {
    out.invertible = b.invertible;
  } else {
    out.invertible = false;
  }

  return out;
}

consteval NodeInfo combine_unary_injective(const NodeInfo &a) {
  NodeInfo out{};
  out.depends_input = a.depends_input;
  out.is_constant = a.is_constant;
  out.constant_value = a.constant_value;
  out.is_affine = false;
  out.invertible = a.invertible;
  return out;
}

consteval NodeInfo combine_neg(const NodeInfo &a) {
  NodeInfo out = combine_unary_injective(a);
  if (a.is_affine) {
    out.is_affine = true;
    out.slope = -a.slope;
    out.intercept = -a.intercept;
  }
  if (a.is_constant)
    out.constant_value = -a.constant_value;
  return out;
}

} // namespace detail_inv

struct InvertibilityResult {
  bool invertible;
  int failing_node;
  OpKind failing_op;
};

template <info Fn> consteval InvertibilityResult invertibility_result() {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());
  constexpr std::size_t N = nodes.size();

  detail_inv::NodeInfo info[N];
  std::size_t input_count = 0;

  template for (constexpr auto n : nodes) {
    if constexpr (n.op == OpKind::Input) {
      ++input_count;
      info[n.self] = detail_inv::make_input();

    } else if constexpr (n.op == OpKind::Const) {
      constexpr double v = static_cast<double>([:n.leaf:]);
      info[n.self] = detail_inv::make_const(v);

    } else if constexpr (n.op == OpKind::Output) {
      info[n.self] = info[n.a];

    } else if constexpr (n.op == OpKind::Add) {
      info[n.self] = detail_inv::combine_add(info[n.a], info[n.b], false);

    } else if constexpr (n.op == OpKind::Sub) {
      info[n.self] = detail_inv::combine_add(info[n.a], info[n.b], true);

    } else if constexpr (n.op == OpKind::Mul) {
      info[n.self] = detail_inv::combine_mul(info[n.a], info[n.b]);

    } else if constexpr (n.op == OpKind::Div) {
      info[n.self] = detail_inv::combine_div(info[n.a], info[n.b]);

    } else if constexpr (n.op == OpKind::Neg) {
      info[n.self] = detail_inv::combine_neg(info[n.a]);

    } else if constexpr (n.op == OpKind::Exp || n.op == OpKind::Log ||
                         n.op == OpKind::Sqrt || n.op == OpKind::Erfc) {
      info[n.self] = detail_inv::combine_unary_injective(info[n.a]);

    } else {
      // Conservative fallback for comparisons, branching, trigonometric and
      // unsupported ops.
      info[n.self] = detail_inv::NodeInfo{};
    }
  }

  if (input_count != 1)
    return {false, -1, OpKind::Input};

  for (std::size_t i = 0; i < N; ++i) {
    if (nodes[i].op == OpKind::Output) {
      if (info[i].invertible)
        return {true, -1, OpKind::Input};
      return {false, static_cast<int>(i), nodes[i].op};
    }
  }

  return {false, -1, OpKind::Output};
}

template <info Fn> consteval bool is_invertible() {
  return invertibility_result<Fn>().invertible;
}

} // namespace ad

#endif // IS_INVERTIBLE_HPP
