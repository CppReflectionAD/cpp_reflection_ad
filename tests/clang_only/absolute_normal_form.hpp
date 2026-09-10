#ifndef ABSOLUTE_NORMAL_FORM_HPP
#define ABSOLUTE_NORMAL_FORM_HPP

// absolute_normal_form.hpp - compile-time rewrite of a piecewise-linear
// function into absolute normal form (ANF).
//
// A piecewise-linear function is one built from its inputs with +, -, scaling
// by a constant, and the kinks abs/max/min. Every such function can be written
// with all of its non-smoothness pushed into a sequence of absolute values
// (Griewank's absolute normal form): one affine map, one abs, repeat.
//
// Griewank's absolute normal form can be written as constant + Ax1 + Bx2 ... +
// a|z1| + b|z2| + ... The x's are input variables, and the z's are called
// switching variables. We get a switching variable at each kink. e.g. given
// input variables u and v
//   max(u, v) = (u + v)/2 + |u - v|/2    - switching variable z1 = u-v
//   min(u, v) = (u + v)/2 - |u - v|/2    - switching variable z1 = u-v
//   relu(u)   = (u + |u|)/2              - switching variable z1 = u
//
// Example:
// y = max(0, x) + min(x, 1-x)
// We can write max(0, x)   = 0.5*( x + |x|)
//              min(x, 1-x) = 0.5*(x + 1-x - |2x-1) = 0.5*(1-|2x-1|)
//
// y = 0.5 + 0.5x + 0.5|x| -0.5 |2x-1|
// This is the absolute normal form. Each max/min has been rewritten as an abs.
// We have introduced new switching variables, z1 = x and z2 = 2x-1. The
// switching coefficients are 0.5 and -0.5 respectively.
//

#include "autograd.h" // for ad::Node, ad::OpKind, ad::build_nodes<>

#include <array>
#include <cstddef>
#include <type_traits>

namespace ad {

// Griewank's absolute normal form can be written as constant + Ax1 + Bx2 ... +
// a|z1| + b|z2| + ... The x's are input variables, and the z's are called
// switching variables. We get a switching variable at each kink. e.g. given
// input variables u and v
template <std::size_t P, std::size_t K> struct AffineForm {
  double constant = 0.0;
  std::array<double, P> input_coeff{};
  std::array<double, K> switch_coeff{};
};

template <std::size_t P, std::size_t K> struct AnfPlan {
  bool ok = false;
  // Where the walk stopped, and on what op (-1 / Input when ok).
  int failing_node = -1;
  OpKind failing_op = OpKind::Input;
  // How many of switching_variables are live; the rest are zero padding.
  std::size_t switching_variable_count = 0;
  // z_i, in dependency order: variable i reads only x and |z_k| for k < i.
  std::array<AffineForm<P, K>, K> switching_variables{};
  AffineForm<P, K> output_form{};
};

namespace detail_anf {

// --- affine arithmetic -----------------------------------------------------

// a + s*b, componentwise.
template <std::size_t P, std::size_t K>
constexpr AffineForm<P, K> combine(const AffineForm<P, K> &a,
                                   const AffineForm<P, K> &b, double s) {
  AffineForm<P, K> out = a;
  out.constant += s * b.constant;
  for (std::size_t i = 0; i < P; ++i)
    out.input_coeff[i] += s * b.input_coeff[i];
  for (std::size_t i = 0; i < K; ++i)
    out.switch_coeff[i] += s * b.switch_coeff[i];
  return out;
}

template <std::size_t P, std::size_t K>
constexpr AffineForm<P, K> scaled(const AffineForm<P, K> &a, double s) {
  AffineForm<P, K> out{};
  out.constant = s * a.constant;
  for (std::size_t i = 0; i < P; ++i)
    out.input_coeff[i] = s * a.input_coeff[i];
  for (std::size_t i = 0; i < K; ++i)
    out.switch_coeff[i] = s * a.switch_coeff[i];
  return out;
}

// No dependence on x or on any |z|: the form is a compile-time number.
template <std::size_t P, std::size_t K>
constexpr bool is_number(const AffineForm<P, K> &a) {
  for (std::size_t i = 0; i < P; ++i)
    if (a.input_coeff[i] != 0.0)
      return false;
  for (std::size_t i = 0; i < K; ++i)
    if (a.switch_coeff[i] != 0.0)
      return false;
  return true;
}

constexpr double magnitude(double x) { return x < 0.0 ? -x : x; }

constexpr bool is_nonzero(double x) { return x != 0.0; }

// Return true if every component of `d` is proportial to the corresponding
// components of `s`. Each component of `d` must be `lambda` * that component of
// `s`. If true, set `lambda` to that constant of proportionality.
template <std::size_t P, std::size_t K>
constexpr bool proportional(const AffineForm<P, K> &d,
                            const AffineForm<P, K> &s, double &lambda) {
  double pivot_d = 0.0;
  double pivot_s = 0.0;

  // Find the first non-zero constant/coefficient in s.
  if (is_nonzero(s.constant)) {
    pivot_d = d.constant;
    pivot_s = s.constant;
  }
  if (!is_nonzero(pivot_s)) {
    for (std::size_t i = 0; i < P; ++i) {
      if (is_nonzero(s.input_coeff[i])) {
        pivot_d = d.input_coeff[i];
        pivot_s = s.input_coeff[i];
        break;
      }
    }
  }
  if (!is_nonzero(pivot_s)) {
    for (std::size_t i = 0; i < K; ++i) {
      if (is_nonzero(s.switch_coeff[i])) {
        pivot_d = d.switch_coeff[i];
        pivot_s = s.switch_coeff[i];
        break;
      }
    }
  }

  // s is identically zero: the condition never switches, so the branches have
  // to be the same function for the Select to be one.
  lambda = is_nonzero(pivot_s) ? pivot_d / pivot_s : 0.0;

  if (d.constant != lambda * s.constant)
    return false;
  for (std::size_t i = 0; i < P; ++i)
    if (d.input_coeff[i] != lambda * s.input_coeff[i])
      return false;
  for (std::size_t i = 0; i < K; ++i)
    if (d.switch_coeff[i] != lambda * s.switch_coeff[i])
      return false;
  return true;
}

// --- shape of the plan, needed before it can be built ----------------------

// Every op that contributes at most one switching variable.
consteval bool is_kink(OpKind op) {
  return op == OpKind::Abs || op == OpKind::Max || op == OpKind::Min ||
         op == OpKind::Relu || op == OpKind::Select;
}

template <info Fn> consteval std::size_t count_inputs() {
  std::size_t p = 0;
  for (const Node &n : build_nodes<Fn>())
    if (n.op == OpKind::Input)
      ++p;
  return p;
}

template <info Fn> consteval std::size_t count_kinks() {
  std::size_t k = 0;
  for (const Node &n : build_nodes<Fn>())
    if (is_kink(n.op))
      ++k;
  return k;
}

// --- the walk --------------------------------------------------------------

template <info Fn>
consteval AnfPlan<count_inputs<Fn>(), count_kinks<Fn>()> build_anf_plan() {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());

  // Work out the shape of the output form.
  constexpr std::size_t N = nodes.size();
  constexpr std::size_t P = count_inputs<Fn>();
  constexpr std::size_t K = count_kinks<Fn>();

  using Form = AffineForm<P, K>;
  using Plan = AnfPlan<P, K>;

  auto fail = [](std::size_t node, OpKind op) {
    Plan p{};
    p.failing_node = static_cast<int>(node);
    p.failing_op = op;
    return p;
  };

  Plan plan{};

  Form form[N];
  // A node carrying a real value. Comparisons and the logical combinators do
  // not; reading one as a number is a failure, not a zero.
  bool usable[N];
  for (std::size_t i = 0; i < N; ++i) {
    form[i] = Form{};
    usable[i] = false;
  }

  struct NewSwitch {
    std::size_t index;
    double scale;
  };
  // Add a switching variable to the output. The swtiching variable, z = s.
  // e.g. if s = 2x1 - 5x2 - 1, then we set z = 2(x2 - 2.5x2 - 0.5). (We scale
  // it so the first coefficient is 1).
  auto push_switch = [&plan](const Form &s) -> NewSwitch {
    // Find the first non-zero coefficient of s.
    double scale = 0.0;
    for (std::size_t i = 0; i < P && !is_nonzero(scale); ++i)
      scale = magnitude(s.input_coeff[i]);
    for (std::size_t i = 0; i < K && !is_nonzero(scale); ++i)
      scale = magnitude(s.switch_coeff[i]);
    if (!is_nonzero(scale))
      scale = 1.0; // s is a number; callers fold those away before getting here

    // Scale s, and add it as a switching variable to the plan.
    const std::size_t index = plan.switching_variable_count++;
    plan.switching_variables[index] = scaled(s, 1.0 / scale);
    return {index, scale};
  };

  // |s| as an affine form; either just a switching variable, or just a
  // constant.
  auto abs_form = [&push_switch](const Form &s) -> Form {
    Form out{};
    if (is_number(s)) {
      out.constant = magnitude(s.constant);
      return out;
    }
    const NewSwitch sw = push_switch(s);
    // Add the switching variable to the form, with coefficient of the
    // calculated scale.
    out.switch_coeff[sw.index] = sw.scale;
    return out;
  };

  bool have_output = false;

  // For each node, create an affine form. Combine affine forms based on the
  // tree structure.
  template for (constexpr auto n : nodes) {
    if constexpr (n.op == OpKind::Input) {
      if constexpr (n.self >= P)
        return fail(n.self, OpKind::Input);
      else {
        // By definition, the input nodes themselves just have coefficient 1 in
        // the affine form. If they are multiplied by a constant, that gets
        // added later.
        form[n.self].input_coeff[n.self] = 1.0;
        usable[n.self] = true;
      }

    } else if constexpr (n.op == OpKind::Const) {
      constexpr double v = static_cast<double>([:n.leaf:]);
      form[n.self].constant = v;
      usable[n.self] = true;

    } else if constexpr (n.op == OpKind::Output) {
      if (!usable[n.a])
        return fail(n.self, OpKind::Output);
      plan.output_form = form[n.a];
      have_output = true;

    } else if constexpr (n.op == OpKind::Add || n.op == OpKind::Sub) {
      if (!usable[n.a] || !usable[n.b])
        return fail(n.self, n.op);
      // Add/substract the two input forms.
      form[n.self] =
          combine(form[n.a], form[n.b], n.op == OpKind::Sub ? -1.0 : 1.0);
      usable[n.self] = true;

    } else if constexpr (n.op == OpKind::Mul) {
      // We can only multiply two forms if at least one is just a number, with
      // no linear terms (otherwise we get a polynomial).
      if (!usable[n.a] || !usable[n.b])
        return fail(n.self, OpKind::Mul);
      if (is_number(form[n.b]))
        form[n.self] = scaled(form[n.a], form[n.b].constant);
      else if (is_number(form[n.a]))
        form[n.self] = scaled(form[n.b], form[n.a].constant);
      else
        return fail(n.self, OpKind::Mul);
      usable[n.self] = true;

    } else if constexpr (n.op == OpKind::Div) {
      // We can only divide a form if the denominator is a pure number.
      if (!usable[n.a] || !usable[n.b])
        return fail(n.self, OpKind::Div);
      if (!is_number(form[n.b]) || !is_nonzero(form[n.b].constant))
        return fail(n.self, OpKind::Div);
      form[n.self] = scaled(form[n.a], 1.0 / form[n.b].constant);
      usable[n.self] = true;

    } else if constexpr (n.op == OpKind::Neg) {
      // Negate every term in the form.
      if (!usable[n.a])
        return fail(n.self, OpKind::Neg);
      form[n.self] = scaled(form[n.a], -1.0);
      usable[n.self] = true;

      // --- the kinks: each is an affine part plus one absolute value --------
    } else if constexpr (n.op == OpKind::Abs) {
      if (!usable[n.a])
        return fail(n.self, OpKind::Abs);
      form[n.self] = abs_form(form[n.a]);
      usable[n.self] = true;

    } else if constexpr (n.op == OpKind::Max || n.op == OpKind::Min) {
      //   max(u, v) = (u + v)/2 + |u - v|/2    - switching variable z1 = u-v
      //   min(u, v) = (u + v)/2 - |u - v|/2    - switching variable z1 = u-v
      if (!usable[n.a] || !usable[n.b])
        return fail(n.self, n.op);
      const Form mid = scaled(combine(form[n.a], form[n.b], 1.0), 0.5);
      const Form diff = combine(form[n.a], form[n.b], -1.0);
      const double half = n.op == OpKind::Min ? -0.5 : 0.5;
      form[n.self] = combine(mid, abs_form(diff), half);
      usable[n.self] = true;

    } else if constexpr (n.op == OpKind::Relu) {
      //   relu(u)   = (u + |u|)/2              - switching variable z1 = u
      if (!usable[n.a])
        return fail(n.self, OpKind::Relu);
      form[n.self] = combine(scaled(form[n.a], 0.5), abs_form(form[n.a]), 0.5);
      usable[n.self] = true;

      // --- conditions carry no value; only a Select may read them -----------
    } else if constexpr (op_is_boolean(n.op)) {
      usable[n.self] = false;

    } else if constexpr (n.op == OpKind::Select) {
      constexpr OpKind cop = nodes[n.cond].op;
      if constexpr (cop != OpKind::Lt && cop != OpKind::Le &&
                    cop != OpKind::Gt && cop != OpKind::Ge)
        // A conjunction/disjunction/equality switches on more than one
        // surface, so it is not a single kink.
        return fail(n.self, OpKind::Select);
      else {
        constexpr std::size_t cu = nodes[n.cond].a;
        constexpr std::size_t cv = nodes[n.cond].b;
        if (!usable[cu] || !usable[cv] || !usable[n.a] || !usable[n.b])
          return fail(n.self, OpKind::Select);

        // s = u - v switches sign at the branch; d = a - b is the jump across
        // it. The branch is a kink rather than a jump iff d = lambda*s, and
        // then the whole Select is b + lambda*min(s, 0) for `<` (the a-branch
        // is the s < 0 side) or b + lambda*max(s, 0) for `>`, using
        //   min(s, 0) = (s - |s|)/2,  max(s, 0) = (s + |s|)/2.
        const Form s = combine(form[cu], form[cv], -1.0);
        const Form d = combine(form[n.a], form[n.b], -1.0);
        double lambda = 0.0;
        if (!proportional(d, s, lambda))
          return fail(n.self, OpKind::Select);

        constexpr double abs_sign =
            (cop == OpKind::Lt || cop == OpKind::Le) ? -1.0 : 1.0;
        Form out = combine(form[n.b], s, 0.5 * lambda);
        out = combine(out, abs_form(s), abs_sign * 0.5 * lambda);
        form[n.self] = out;
        usable[n.self] = true;
      }

    } else {
      // Anything smooth-but-not-affine (exp, log, sin, ...) or a tensor op.
      return fail(n.self, n.op);
    }
  }

  if (!have_output)
    return fail(0, OpKind::Output);

  plan.ok = true;
  plan.failing_node = -1;
  return plan;
}

// Evaluate one switching variable. The switching variable has the form
// `switch_form`. Evaluate it using input variables `x` and switching variables
// `abs_z`.
template <typename T, std::size_t P, std::size_t K>
constexpr T eval_switch(const AffineForm<P, K> &switch_form,
                        const std::array<T, P> &x,
                        const std::array<T, K> &abs_z) {
  T acc = static_cast<T>(switch_form.constant);
  for (std::size_t i = 0; i < P; ++i)
    acc += static_cast<T>(switch_form.input_coeff[i]) * x[i];
  for (std::size_t i = 0; i < K; ++i)
    acc += static_cast<T>(switch_form.switch_coeff[i]) * abs_z[i];
  return acc;
}

} // namespace detail_anf

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

struct AnfResult {
  bool representable;
  int failing_node;
  OpKind failing_op;
  // Number of absolute values in the form (0 for an affine function).
  std::size_t switching_variable_count;
};

template <info Fn> consteval AnfResult anf_result() {
  constexpr auto plan = detail_anf::build_anf_plan<Fn>();
  return {plan.ok, plan.failing_node, plan.failing_op,
          plan.switching_variable_count};
}

template <info Fn> consteval bool has_absolute_normal_form() {
  return anf_result<Fn>().representable;
}

// The function rewritten into ANF: same values as Fn, evaluated as a chain of
// affine input variables and absolute values. `plan` holds the coefficients.
template <info Fn> struct absolute_normal_form {
  static_assert(has_absolute_normal_form<Fn>(),
                "ad::absolute_normal_form requires a piecewise-linear "
                "function; when no absolute normal form can be built, "
                "ad::has_absolute_normal_form<Fn>() is false");

  static constexpr auto plan = detail_anf::build_anf_plan<Fn>();

  static constexpr std::size_t input_count =
      plan.output_form.input_coeff.size();
  static constexpr std::size_t switching_variable_count =
      plan.switching_variable_count;

  template <typename... Ts>
  constexpr std::common_type_t<Ts..., double> operator()(Ts... xs) const {
    static_assert(sizeof...(Ts) == input_count,
                  "ad::absolute_normal_form takes one argument per parameter "
                  "of the reflected function");
    using T = std::common_type_t<Ts..., double>;

    // Create an array to store each input variable.
    const std::array<T, input_count> x{static_cast<T>(xs)...};
    // And an array to store each switching variable.
    std::array<T, plan.switching_variables.size()> abs_z{};
    // Iterate through each switching variable we want to add.
    for (std::size_t i = 0; i < plan.switching_variable_count; ++i) {
      // Evaluate it the switching variable.
      const T z =
          detail_anf::eval_switch(plan.switching_variables[i], x, abs_z);
      // Work out it's absolute value so we can add it to our array.
      abs_z[i] = z < static_cast<T>(0) ? -z : z;
    }
    // Now we have all switching variables, work out the actualy affine result.
    return detail_anf::eval_switch(plan.output_form, x, abs_z);
  }
};

template <info Fn, typename... Ts>
constexpr std::common_type_t<Ts..., double> absolute_normal_form_of(Ts... xs) {
  return absolute_normal_form<Fn>{}(xs...);
}

} // namespace ad

#endif // ABSOLUTE_NORMAL_FORM_HPP
