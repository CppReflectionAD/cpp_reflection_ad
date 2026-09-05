#ifndef IS_INVERTIBLE_HPP
#define IS_INVERTIBLE_HPP

// is_invertible.hpp - compile-time explicit inverse checker for AD.
//
// This module only reports invertible when it can build an explicit inverse
// expression (as a reversible step plan). No numeric root finder fallback is
// used: if no explicit inverse plan is found, is_invertible returns false.

#include "autograd.h" // for ad::Node, ad::OpKind, ad::build_nodes<>

#include <array>
#include <cmath>
#include <cstddef>

namespace ad {

namespace detail_inv {

struct NodeInfo {
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
  }
  return out;
}

consteval NodeInfo combine_mul(const NodeInfo &a, const NodeInfo &b) {
  NodeInfo out{};
  out.depends_input = a.depends_input || b.depends_input;
  out.is_constant = a.is_constant && b.is_constant;
  if (out.is_constant)
    out.constant_value = a.constant_value * b.constant_value;

  if (a.is_constant && b.is_affine) {
    out.is_affine = true;
    out.slope = a.constant_value * b.slope;
    out.intercept = a.constant_value * b.intercept;
  } else if (b.is_constant && a.is_affine) {
    out.is_affine = true;
    out.slope = b.constant_value * a.slope;
    out.intercept = b.constant_value * a.intercept;
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
  }
  return out;
}

enum class InverseStepKind {
  AddConst,
  SubConst,
  ConstMinus,
  MulConst,
  DivConst,
  ConstDiv,
  Neg,
  Exp,
  Log,
  Sqrt,
};

struct InverseStep {
  InverseStepKind kind = InverseStepKind::AddConst;
  double constant_value = 0.0;
};

template <typename T>
constexpr T apply_inverse_step(const InverseStep &step, T y) {
  switch (step.kind) {
  case InverseStepKind::AddConst:
    return y - static_cast<T>(step.constant_value);
  case InverseStepKind::SubConst:
    return y + static_cast<T>(step.constant_value);
  case InverseStepKind::ConstMinus:
    return static_cast<T>(step.constant_value) - y;
  case InverseStepKind::MulConst:
    return y / static_cast<T>(step.constant_value);
  case InverseStepKind::DivConst:
    return y * static_cast<T>(step.constant_value);
  case InverseStepKind::ConstDiv:
    return static_cast<T>(step.constant_value) / y;
  case InverseStepKind::Neg:
    return -y;
  case InverseStepKind::Exp:
    return std::log(y);
  case InverseStepKind::Log:
    return std::exp(y);
  case InverseStepKind::Sqrt:
    return y * y;
  }
  return y;
}

template <info Fn> consteval auto build_inverse_plan() {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());
  constexpr std::size_t N = nodes.size();

  struct InversePlan {
    bool ok;
    int failing_node;
    OpKind failing_op;
    std::size_t step_count;
    std::array<InverseStep, N> steps;
  };

  auto fail = [](int node, OpKind op) {
    return InversePlan{false, node, op, 0, {}};
  };

  NodeInfo info[N];
  for (std::size_t i = 0; i < N; ++i)
    info[i] = NodeInfo{};

  std::size_t input_count = 0;
  int output_idx = -1;

  template for (constexpr auto n : nodes) {
    if constexpr (n.op == OpKind::Input) {
      ++input_count;
      info[n.self] = make_input();
    } else if constexpr (n.op == OpKind::Const) {
      constexpr double v = static_cast<double>([:n.leaf:]);
      info[n.self] = make_const(v);
    } else if constexpr (n.op == OpKind::Output) {
      output_idx = static_cast<int>(n.self);
      info[n.self] = info[n.a];
    } else if constexpr (n.op == OpKind::Add) {
      info[n.self] = combine_add(info[n.a], info[n.b], false);
    } else if constexpr (n.op == OpKind::Sub) {
      info[n.self] = combine_add(info[n.a], info[n.b], true);
    } else if constexpr (n.op == OpKind::Mul) {
      info[n.self] = combine_mul(info[n.a], info[n.b]);
    } else if constexpr (n.op == OpKind::Div) {
      info[n.self] = combine_div(info[n.a], info[n.b]);
    } else if constexpr (n.op == OpKind::Neg || n.op == OpKind::Exp ||
                         n.op == OpKind::Log || n.op == OpKind::Sqrt ||
                         n.op == OpKind::Erfc) {
      info[n.self].depends_input = info[n.a].depends_input;
      info[n.self].is_constant = info[n.a].is_constant;
      info[n.self].constant_value = info[n.a].constant_value;
    } else {
      info[n.self] = NodeInfo{};
    }
  }

  if (input_count != 1)
    return fail(-1, OpKind::Input);

  if (output_idx < 0)
    return fail(-1, OpKind::Output);

  const auto &out = nodes[static_cast<std::size_t>(output_idx)];
  if (out.op != OpKind::Output)
    return fail(output_idx, out.op);

  if (!info[out.a].depends_input)
    return fail(output_idx, OpKind::Output);

  std::array<InverseStep, N> steps{};
  std::size_t step_count = 0;
  std::size_t cur = out.a;

  while (true) {
    const auto &n = nodes[cur];

    if (n.op == OpKind::Input)
      break;

    if (n.op == OpKind::Neg) {
      steps[step_count++] = InverseStep{InverseStepKind::Neg, 0.0};
      cur = n.a;
      continue;
    }

    if (n.op == OpKind::Exp) {
      steps[step_count++] = InverseStep{InverseStepKind::Exp, 0.0};
      cur = n.a;
      continue;
    }

    if (n.op == OpKind::Log) {
      steps[step_count++] = InverseStep{InverseStepKind::Log, 0.0};
      cur = n.a;
      continue;
    }

    if (n.op == OpKind::Sqrt) {
      steps[step_count++] = InverseStep{InverseStepKind::Sqrt, 0.0};
      cur = n.a;
      continue;
    }

    if (n.op == OpKind::Erfc)
      return fail(static_cast<int>(cur), n.op);

    if (n.op == OpKind::Add || n.op == OpKind::Sub || n.op == OpKind::Mul ||
        n.op == OpKind::Div) {
      const bool a_dep = info[n.a].depends_input;
      const bool b_dep = info[n.b].depends_input;
      if (a_dep == b_dep)
        return fail(static_cast<int>(cur), n.op);

      const std::size_t dep_idx = a_dep ? n.a : n.b;
      const std::size_t cst_idx = a_dep ? n.b : n.a;
      if (!info[cst_idx].is_constant)
        return fail(static_cast<int>(cur), n.op);

      const double c = info[cst_idx].constant_value;

      if (n.op == OpKind::Add) {
        steps[step_count++] = InverseStep{InverseStepKind::AddConst, c};
      } else if (n.op == OpKind::Sub) {
        steps[step_count++] = a_dep
                                  ? InverseStep{InverseStepKind::SubConst, c}
                                  : InverseStep{InverseStepKind::ConstMinus, c};
      } else if (n.op == OpKind::Mul) {
        if (!is_nonzero(c))
          return fail(static_cast<int>(cur), n.op);
        steps[step_count++] = InverseStep{InverseStepKind::MulConst, c};
      } else {
        if (a_dep) {
          if (!is_nonzero(c))
            return fail(static_cast<int>(cur), n.op);
          steps[step_count++] = InverseStep{InverseStepKind::DivConst, c};
        } else {
          if (!is_nonzero(c))
            return fail(static_cast<int>(cur), n.op);
          steps[step_count++] = InverseStep{InverseStepKind::ConstDiv, c};
        }
      }

      cur = dep_idx;
      continue;
    }

    return fail(static_cast<int>(cur), n.op);
  }

  return InversePlan{true, -1, OpKind::Input, step_count, steps};
}

} // namespace detail_inv

struct InvertibilityResult {
  bool invertible;
  int failing_node;
  OpKind failing_op;
};

template <info Fn> consteval InvertibilityResult invertibility_result() {
  constexpr auto plan = detail_inv::build_inverse_plan<Fn>();
  return {plan.ok, plan.failing_node, plan.failing_op};
}

template <info Fn> consteval bool is_invertible() {
  return invertibility_result<Fn>().invertible;
}

template <info Fn> struct inverse {
  static_assert(
      is_invertible<Fn>(),
      "ad::inverse requires an explicit inverse plan; if the inverse "
      "is not constructible symbolically, ad::is_invertible<Fn>() is false");

  template <typename T = double> constexpr T operator()(T y) const {
    constexpr auto plan = detail_inv::build_inverse_plan<Fn>();
    T x = y;
    for (std::size_t i = 0; i < plan.step_count; ++i)
      x = detail_inv::apply_inverse_step(plan.steps[i], x);
    return x;
  }
};

template <info Fn, typename T = double> constexpr T inverse_of(T y) {
  return inverse<Fn>{}(y);
}

} // namespace ad

#endif // IS_INVERTIBLE_HPP
