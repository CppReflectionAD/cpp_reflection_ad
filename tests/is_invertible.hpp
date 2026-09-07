#ifndef IS_INVERTIBLE_HPP
#define IS_INVERTIBLE_HPP

// is_invertible.hpp - compile-time explicit inverse checker for AD.
//
// This module only reports invertible when it can build an explicit inverse
// expression (as a reversible step plan). No numeric root finder fallback is
// used: if no explicit inverse plan is found, is_invertible returns false.

#include "autograd.h" // for ad::Node, ad::OpKind, ad::build_nodes<>
#include "mc_sim/normal_distribution.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace ad {

template <info Fn, auto InvFn> struct inverse_pair {
  static constexpr auto function = Fn;
  static constexpr auto inverse = InvFn;
};

template <info Fn, std::size_t ArgIndex, auto InvFn> struct inverse_pair_wrt {
  static constexpr auto function = Fn;
  static constexpr std::size_t arg_index = ArgIndex;
  static constexpr auto inverse = InvFn;
};

using default_cdf_pair = inverse_pair<^^mcsim::CDF, mcsim::CDF_inverse>;
using default_cdf_inverse_pair = inverse_pair<^^mcsim::CDF_inverse, mcsim::CDF>;

template <info QueryFn, typename Pair> struct pair_matches : std::false_type {};

template <typename Pair> struct is_unary_inverse_pair : std::false_type {};

template <info QueryFn, auto Candidate> consteval bool query_fn_value_equals() {
  constexpr auto query_value = [:QueryFn:];
  if constexpr (std::is_same_v<decltype(query_value), decltype(Candidate)>)
    return query_value == Candidate;
  else
    return false;
}

template <info QueryFn, info Fn, auto InvFn>
struct pair_matches<QueryFn, inverse_pair<Fn, InvFn>>
    : std::bool_constant<(QueryFn == Fn ||
                          query_fn_value_equals<QueryFn, InvFn>())> {};

template <info Fn, auto InvFn>
struct is_unary_inverse_pair<inverse_pair<Fn, InvFn>> : std::true_type {};

template <info QueryFn, std::size_t QueryArgIndex, typename Pair>
struct pair_matches_wrt : std::false_type {};

template <info QueryFn, std::size_t QueryArgIndex, info Fn,
          std::size_t ArgIndex, auto InvFn>
struct pair_matches_wrt<QueryFn, QueryArgIndex,
                        inverse_pair_wrt<Fn, ArgIndex, InvFn>>
    : std::bool_constant<(QueryFn == Fn && QueryArgIndex == ArgIndex)> {};

template <info QueryFn, typename... Pairs>
consteval bool has_registered_inverse() {
  return (pair_matches<QueryFn, Pairs>::value || ...);
}

template <info QueryFn, std::size_t QueryArgIndex, typename... Pairs>
consteval bool has_registered_inverse_wrt() {
  return (pair_matches_wrt<QueryFn, QueryArgIndex, Pairs>::value || ...);
}

template <typename... Pairs> consteval bool has_unary_inverse_pair() {
  return (is_unary_inverse_pair<Pairs>::value || ...);
}

template <typename T, info Fn, auto InvFn>
constexpr T apply_pair_inverse(inverse_pair<Fn, InvFn>, T y) {
  if constexpr (requires { InvFn(y); })
    return static_cast<T>(InvFn(y));
  else
    return static_cast<T>(InvFn(static_cast<double>(y)));
}

template <typename T, info Fn, auto InvFn>
constexpr T apply_pair_forward(inverse_pair<Fn, InvFn>, T y) {
  if constexpr (requires { [:Fn:](y); })
    return static_cast<T>([:Fn:](y));
  else
    return static_cast<T>([:Fn:](static_cast<double>(y)));
}

template <info QueryFn, typename T, info Fn, auto InvFn>
constexpr T apply_pair_inverse_for(inverse_pair<Fn, InvFn>, T y) {
  if constexpr (QueryFn == Fn)
    return apply_pair_inverse(inverse_pair<Fn, InvFn>{}, y);
  else if constexpr (query_fn_value_equals<QueryFn, InvFn>())
    return apply_pair_forward(inverse_pair<Fn, InvFn>{}, y);
  else {
    static_assert(QueryFn == Fn || query_fn_value_equals<QueryFn, InvFn>(),
                  "Pair does not match queried function");
    return T{};
  }
}

template <typename T, info Fn, std::size_t ArgIndex, auto InvFn,
          typename... ExtraArgs>
constexpr T apply_pair_inverse_wrt(inverse_pair_wrt<Fn, ArgIndex, InvFn>, T y,
                                   ExtraArgs... args) {
  if constexpr (requires { InvFn(y, args...); })
    return static_cast<T>(InvFn(y, args...));
  else
    return static_cast<T>(
        InvFn(static_cast<double>(y), static_cast<double>(args)...));
}

template <info Fn, typename T, typename FirstPair, typename... RestPairs>
constexpr T apply_registered_inverse(T y) {
  if constexpr (pair_matches<Fn, FirstPair>::value)
    return apply_pair_inverse_for<Fn>(FirstPair{}, y);
  else
    return apply_registered_inverse<Fn, T, RestPairs...>(y);
}

template <info Fn, typename T> constexpr T apply_registered_inverse(T) {
  static_assert(Fn != Fn,
                "No inverse pair registered for this function in this call");
  return T{};
}

template <typename T> constexpr T finite_abs_or_inf(T expected, T actual) {
  const T err = std::abs(actual - expected);
  if (std::isfinite(err))
    return err;
  return std::numeric_limits<T>::infinity();
}

template <bool UseInverseDirection, typename T, typename FirstPair,
          typename... RestPairs>
constexpr T apply_registered_unary_transform(T y) {
  if constexpr (is_unary_inverse_pair<FirstPair>::value) {
    if constexpr (UseInverseDirection)
      return apply_pair_inverse(FirstPair{}, y);
    else
      return apply_pair_forward(FirstPair{}, y);
  } else {
    return apply_registered_unary_transform<UseInverseDirection, T,
                                            RestPairs...>(y);
  }
}

template <bool UseInverseDirection, typename T>
constexpr T apply_registered_unary_transform(T) {
  static_assert(!std::is_same_v<T, T>,
                "No unary inverse pair available to unwrap output");
  return T{};
}

template <typename T, typename FirstPair, typename... RestPairs>
constexpr bool choose_unary_unwrap_inverse_direction(T y, T b_wrapped,
                                                     T one_wrapped) {
  if constexpr (is_unary_inverse_pair<FirstPair>::value) {
    const T inv_y = apply_pair_inverse(FirstPair{}, y);
    const T inv_b = apply_pair_inverse(FirstPair{}, b_wrapped);
    const T inv_one = apply_pair_inverse(FirstPair{}, one_wrapped);
    const T inv_err =
        finite_abs_or_inf(y, apply_pair_forward(FirstPair{}, inv_y)) +
        finite_abs_or_inf(b_wrapped, apply_pair_forward(FirstPair{}, inv_b)) +
        finite_abs_or_inf(one_wrapped,
                          apply_pair_forward(FirstPair{}, inv_one));

    const T fwd_y = apply_pair_forward(FirstPair{}, y);
    const T fwd_b = apply_pair_forward(FirstPair{}, b_wrapped);
    const T fwd_one = apply_pair_forward(FirstPair{}, one_wrapped);
    const T fwd_err =
        finite_abs_or_inf(y, apply_pair_inverse(FirstPair{}, fwd_y)) +
        finite_abs_or_inf(b_wrapped, apply_pair_inverse(FirstPair{}, fwd_b)) +
        finite_abs_or_inf(one_wrapped,
                          apply_pair_inverse(FirstPair{}, fwd_one));

    return inv_err <= fwd_err;
  } else {
    return choose_unary_unwrap_inverse_direction<T, RestPairs...>(y, b_wrapped,
                                                                  one_wrapped);
  }
}

template <typename T>
constexpr bool choose_unary_unwrap_inverse_direction(T, T, T) {
  static_assert(!std::is_same_v<T, T>,
                "No unary inverse pair available to unwrap output");
  return true;
}

template <info Fn, std::size_t ArgIndex, typename T, typename FirstPair,
          typename... RestPairs, typename... ExtraArgs>
constexpr T apply_registered_inverse_wrt(T y, ExtraArgs... args) {
  if constexpr (pair_matches_wrt<Fn, ArgIndex, FirstPair>::value)
    return apply_pair_inverse_wrt(FirstPair{}, y, args...);
  else
    return apply_registered_inverse_wrt<Fn, ArgIndex, T, RestPairs...>(y,
                                                                       args...);
}

template <info Fn, std::size_t ArgIndex, typename T, typename... ExtraArgs>
constexpr T apply_registered_inverse_wrt(T, ExtraArgs...) {
  static_assert(Fn != Fn,
                "No partial inverse pair registered for this function/arg "
                "index in this call");
  return T{};
}

namespace detail_inv {

template <info Fn> consteval std::size_t input_count_of() {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());
  std::size_t count = 0;
  template for (constexpr auto n : nodes) {
    if constexpr (n.op == OpKind::Input)
      ++count;
  }
  return count;
}

template <info Fn, std::size_t ArgIndex> consteval bool is_affine_in_arg() {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());
  constexpr std::size_t N = nodes.size();
  constexpr std::size_t InputCount = input_count_of<Fn>();
  if constexpr (ArgIndex >= InputCount)
    return false;

  struct WrtInfo {
    bool depends_target = false;
    bool affine_target = false;
  };

  WrtInfo info[N];
  for (std::size_t i = 0; i < N; ++i)
    info[i] = WrtInfo{};

  int output_idx = -1;

  template for (constexpr auto n : nodes) {
    if constexpr (n.op == OpKind::Input) {
      info[n.self].depends_target = (n.self == ArgIndex);
      info[n.self].affine_target = true;
    } else if constexpr (n.op == OpKind::Const) {
      info[n.self].depends_target = false;
      info[n.self].affine_target = true;
    } else if constexpr (n.op == OpKind::Output) {
      output_idx = static_cast<int>(n.self);
      info[n.self] = info[n.a];
    } else if constexpr (n.op == OpKind::Add || n.op == OpKind::Sub) {
      info[n.self].depends_target =
          info[n.a].depends_target || info[n.b].depends_target;
      info[n.self].affine_target =
          info[n.a].affine_target && info[n.b].affine_target;
    } else if constexpr (n.op == OpKind::Mul) {
      const bool a_dep = info[n.a].depends_target;
      const bool b_dep = info[n.b].depends_target;
      info[n.self].depends_target = a_dep || b_dep;
      info[n.self].affine_target = info[n.a].affine_target &&
                                   info[n.b].affine_target && !(a_dep && b_dep);
    } else if constexpr (n.op == OpKind::Div) {
      const bool a_dep = info[n.a].depends_target;
      const bool b_dep = info[n.b].depends_target;
      info[n.self].depends_target = a_dep || b_dep;
      info[n.self].affine_target =
          info[n.a].affine_target && info[n.b].affine_target && !b_dep;
    } else if constexpr (n.op == OpKind::Neg) {
      info[n.self].depends_target = info[n.a].depends_target;
      info[n.self].affine_target = info[n.a].affine_target;
    } else if constexpr (n.op == OpKind::Exp || n.op == OpKind::Log ||
                         n.op == OpKind::Sqrt || n.op == OpKind::Erfc ||
                         n.op == OpKind::Sin || n.op == OpKind::Cos ||
                         n.op == OpKind::Abs) {
      info[n.self].depends_target = info[n.a].depends_target;
      info[n.self].affine_target = !info[n.a].depends_target;
    } else {
      // Conservative fallback for unsupported ops.
      bool dep = false;
      bool aff = true;
      if constexpr (op_has_a(n.op)) {
        dep = dep || info[n.a].depends_target;
        aff = aff && info[n.a].affine_target;
      }
      if constexpr (op_has_b(n.op)) {
        dep = dep || info[n.b].depends_target;
        aff = aff && info[n.b].affine_target;
      }
      if constexpr (op_has_cond(n.op))
        dep = dep || info[n.cond].depends_target;
      info[n.self].depends_target = dep;
      info[n.self].affine_target = aff && !dep;
    }
  }

  if (output_idx < 0)
    return false;

  const auto &out = nodes[static_cast<std::size_t>(output_idx)];
  if (out.op != OpKind::Output)
    return false;

  return info[out.a].depends_target && info[out.a].affine_target;
}

template <info Fn, std::size_t ArgIndex, typename T, typename... ExtraArgs>
constexpr T eval_with_arg(T target_x, ExtraArgs... extras) {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());
  constexpr std::size_t N = nodes.size();
  constexpr std::size_t InputCount = input_count_of<Fn>();
  static_assert(sizeof...(ExtraArgs) + 1 == InputCount,
                "inverse_wrt expects all non-target arguments");

  T in[InputCount] = {};
  const T extra_vals[] = {static_cast<T>(extras)...};
  std::size_t extra_i = 0;
  for (std::size_t i = 0; i < InputCount; ++i) {
    if (i == ArgIndex)
      in[i] = target_x;
    else
      in[i] = extra_vals[extra_i++];
  }

  T val[N] = {};
  template for (constexpr auto n : nodes) {
    if constexpr (n.op == OpKind::Input)
      val[n.self] = in[n.self];
    else if constexpr (n.op == OpKind::Const)
      val[n.self] = static_cast<T>([:n.leaf:]);
    else if constexpr (n.op == OpKind::Output)
      val[n.self] = val[n.a];
    else if constexpr (n.op == OpKind::Add)
      val[n.self] = val[n.a] + val[n.b];
    else if constexpr (n.op == OpKind::Sub)
      val[n.self] = val[n.a] - val[n.b];
    else if constexpr (n.op == OpKind::Mul)
      val[n.self] = val[n.a] * val[n.b];
    else if constexpr (n.op == OpKind::Div)
      val[n.self] = val[n.a] / val[n.b];
    else if constexpr (n.op == OpKind::Neg)
      val[n.self] = -val[n.a];
    else if constexpr (n.op == OpKind::Sin)
      val[n.self] = std::sin(val[n.a]);
    else if constexpr (n.op == OpKind::Cos)
      val[n.self] = std::cos(val[n.a]);
    else if constexpr (n.op == OpKind::Exp)
      val[n.self] = std::exp(val[n.a]);
    else if constexpr (n.op == OpKind::Log)
      val[n.self] = std::log(val[n.a]);
    else if constexpr (n.op == OpKind::Sqrt)
      val[n.self] = std::sqrt(val[n.a]);
    else if constexpr (n.op == OpKind::Erfc)
      val[n.self] = std::erfc(val[n.a]);
    else
      val[n.self] = T{};
  }

  return val[N - 1];
}

template <info Fn, std::size_t ArgIndex, typename T, typename... ExtraArgs>
constexpr T eval_fn_with_arg_runtime(T target_x, ExtraArgs... extras) {
  constexpr std::size_t InputCount = sizeof...(ExtraArgs) + 1;
  T in[InputCount] = {};
  const T extra_vals[] = {static_cast<T>(extras)...};
  std::size_t extra_i = 0;
  for (std::size_t i = 0; i < InputCount; ++i) {
    if (i == ArgIndex)
      in[i] = target_x;
    else
      in[i] = extra_vals[extra_i++];
  }

  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    if constexpr (requires { [:Fn:](in[I]...); })
      return static_cast<T>([:Fn:](in[I]...));
    else
      return static_cast<T>([:Fn:](static_cast<double>(in[I])...));
  }(std::make_index_sequence<InputCount>{});
}

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

template <info Fn, typename... RegisteredPairs>
consteval auto build_inverse_plan() {
  constexpr std::size_t PlanStepCapacity = [] {
    if constexpr (has_registered_inverse<Fn, RegisteredPairs...>()) {
      return std::size_t{56};
    } else {
      static constexpr auto nodes_for_size =
          std::define_static_array(build_nodes<Fn>());
      return nodes_for_size.size();
    }
  }();

  struct InversePlan {
    bool ok;
    int failing_node;
    OpKind failing_op;
    std::size_t step_count;
    std::array<InverseStep, PlanStepCapacity> steps;
  };

  auto fail = [](int node, OpKind op) {
    return InversePlan{false, node, op, 0, {}};
  };

  if constexpr (has_registered_inverse<Fn, RegisteredPairs...>()) {
    return InversePlan{true, -1, OpKind::Input, 0, {}};
  } else {
    static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());
    constexpr std::size_t N = nodes.size();

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
        if constexpr (n.op == OpKind::Neg) {
          if (info[n.a].is_affine) {
            info[n.self].is_affine = true;
            info[n.self].slope = -info[n.a].slope;
            info[n.self].intercept = -info[n.a].intercept;
          }
        }
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

    // If the whole function is affine with nonzero slope, we can always emit an
    // explicit symbolic inverse: x = (y - b) / a.
    if (info[out.a].is_affine && is_nonzero(info[out.a].slope)) {
      std::array<InverseStep, N> affine_steps{};
      std::size_t affine_step_count = 0;
      affine_steps[affine_step_count++] =
          InverseStep{InverseStepKind::AddConst, info[out.a].intercept};
      affine_steps[affine_step_count++] =
          InverseStep{InverseStepKind::MulConst, info[out.a].slope};
      return InversePlan{true, -1, OpKind::Input, affine_step_count,
                         affine_steps};
    }

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
          steps[step_count++] =
              a_dep ? InverseStep{InverseStepKind::SubConst, c}
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
}

} // namespace detail_inv

struct InvertibilityResult {
  bool invertible;
  int failing_node;
  OpKind failing_op;
};

template <info Fn, typename... RegisteredPairs>
consteval InvertibilityResult invertibility_result() {
  constexpr auto plan =
      detail_inv::build_inverse_plan<Fn, RegisteredPairs..., default_cdf_pair,
                                     default_cdf_inverse_pair>();
  return {plan.ok, plan.failing_node, plan.failing_op};
}

template <info Fn, typename... RegisteredPairs> consteval bool is_invertible() {
  return invertibility_result<Fn, RegisteredPairs...>().invertible;
}

template <info Fn, std::size_t ArgIndex, typename... RegisteredPairs>
consteval InvertibilityResult invertibility_result_wrt() {
  if constexpr (has_registered_inverse_wrt<Fn, ArgIndex, RegisteredPairs...>())
    return {true, -1, OpKind::Input};
  else if constexpr (has_unary_inverse_pair<RegisteredPairs...>())
    return {true, -1, OpKind::Input};
  else if constexpr (detail_inv::is_affine_in_arg<Fn, ArgIndex>())
    return {true, -1, OpKind::Input};
  else
    return {false, -1, OpKind::Input};
}

template <info Fn, std::size_t ArgIndex, typename... RegisteredPairs>
consteval bool is_invertible_wrt() {
  return invertibility_result_wrt<Fn, ArgIndex, RegisteredPairs...>()
      .invertible;
}

template <info Fn, typename... RegisteredPairs> struct inverse {
  static_assert(
      is_invertible<Fn, RegisteredPairs...>(),
      "ad::inverse requires an explicit inverse plan; if the inverse "
      "is not constructible symbolically, ad::is_invertible<Fn>() is false");

  template <typename T = double> constexpr T operator()(T y) const {
    if constexpr (has_registered_inverse<Fn, RegisteredPairs...,
                                         default_cdf_pair,
                                         default_cdf_inverse_pair>()) {
      return apply_registered_inverse<Fn, T, RegisteredPairs...,
                                      default_cdf_pair,
                                      default_cdf_inverse_pair>(y);
    }

    constexpr auto plan =
        detail_inv::build_inverse_plan<Fn, RegisteredPairs..., default_cdf_pair,
                                       default_cdf_inverse_pair>();
    T x = y;
    for (std::size_t i = 0; i < plan.step_count; ++i)
      x = detail_inv::apply_inverse_step(plan.steps[i], x);
    return x;
  }
};

template <info Fn, std::size_t ArgIndex, typename... RegisteredPairs>
struct inverse_wrt {
  static_assert(
      is_invertible_wrt<Fn, ArgIndex, RegisteredPairs...>(),
      "ad::inverse_wrt requires either: (1) a registered partial inverse "
      "pair, (2) an affine-in-argument function, or (3) a registered unary "
      "outer inverse pair");

  template <typename T = double, typename... ExtraArgs>
  constexpr T operator()(T y, ExtraArgs... args) const {
    if constexpr (has_registered_inverse_wrt<Fn, ArgIndex,
                                             RegisteredPairs...>()) {
      return apply_registered_inverse_wrt<Fn, ArgIndex, T, RegisteredPairs...>(
          y, args...);
    } else if constexpr (has_unary_inverse_pair<RegisteredPairs...>()) {
      // Handles y = g(a*x + b), where g^{-1} is given by a registered unary
      // inverse_pair. Unwrap g first, then solve the affine map.
      const T b_wrapped =
          detail_inv::eval_fn_with_arg_runtime<Fn, ArgIndex, T>(T{0}, args...);
      const T one_wrapped =
          detail_inv::eval_fn_with_arg_runtime<Fn, ArgIndex, T>(T{1}, args...);

      const bool use_inverse_direction =
          choose_unary_unwrap_inverse_direction<T, RegisteredPairs...>(
              y, b_wrapped, one_wrapped);

      const T y_unwrapped =
          use_inverse_direction
              ? apply_registered_unary_transform<true, T, RegisteredPairs...>(y)
              : apply_registered_unary_transform<false, T, RegisteredPairs...>(
                    y);
      const T b =
          use_inverse_direction
              ? apply_registered_unary_transform<true, T, RegisteredPairs...>(
                    b_wrapped)
              : apply_registered_unary_transform<false, T, RegisteredPairs...>(
                    b_wrapped);
      const T one_unwrapped =
          use_inverse_direction
              ? apply_registered_unary_transform<true, T, RegisteredPairs...>(
                    one_wrapped)
              : apply_registered_unary_transform<false, T, RegisteredPairs...>(
                    one_wrapped);
      const T a = one_unwrapped - b;
      return (y_unwrapped - b) / a;
    } else {
      const T b = detail_inv::eval_with_arg<Fn, ArgIndex, T>(T{0}, args...);
      const T a = detail_inv::eval_with_arg<Fn, ArgIndex, T>(T{1}, args...) - b;
      return (y - b) / a;
    }
  }
};

template <info Fn, typename T = double, typename... RegisteredPairs>
constexpr T inverse_of(T y) {
  return inverse<Fn, RegisteredPairs...>{}(y);
}

template <info Fn, std::size_t ArgIndex, typename T = double,
          typename... RegisteredPairs, typename... ExtraArgs>
constexpr T inverse_of_wrt(T y, ExtraArgs... args) {
  return inverse_wrt<Fn, ArgIndex, RegisteredPairs...>{}(y, args...);
}

} // namespace ad

#endif // IS_INVERTIBLE_HPP
