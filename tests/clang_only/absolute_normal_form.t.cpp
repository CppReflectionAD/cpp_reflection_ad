#include "absolute_normal_form.hpp"
#include <test_simple_include.hpp>

#include <cmath>

// ---------------------------------------------------------------------------
// Functions used by the ANF tests
// ---------------------------------------------------------------------------

// The worked example: max(3x, x+4) is 2x + 2 + |x - 2|.
inline double fn_max_example(double x) { return std::fmax(3.0 * x, x + 4.0); }

inline double fn_affine(double x) { return 2.0 * x + 3.0; }
inline double fn_abs_shift(double x) { return 2.0 * std::fabs(x - 1.0) + 3.0; }
inline double fn_min_example(double x) { return std::fmin(3.0 * x, x + 4.0); }
inline double fn_max_two_args(double x, double y) { return std::fmax(x, y); }
// Nested kinks: the second switching row reads the first one's |z|.
inline double fn_clamp(double x) {
  return std::fmin(std::fmax(2.0 * x, -1.0), 2.0);
}
inline double fn_abs_of_abs(double x) { return std::fabs(std::fabs(x) - 1.0); }
// abs of a compile-time number folds away; no switching row is needed.
inline double fn_abs_const(double x) { return x + std::fabs(-3.0); }
// A branch that is a kink rather than a jump: relu, written by hand.
inline double fn_relu(double x) { return x > 0.0 ? x : 0.0; }
inline double fn_kinked_branch(double x) {
  return x < 1.0 ? 2.0 * x : 3.0 * x - 1.0;
}

// Not piecewise linear (or not provably so).
inline double fn_square(double x) { return x * x; }
inline double fn_exp(double x) { return std::exp(x); }
inline double fn_ratio(double x, double y) { return x / y; }
// Continuous, but only after a cancellation the syntactic walk cannot see.
inline double fn_cancelling(double x) { return x * x / x; }
// A branch that really jumps.
inline double fn_step(double x) { return x > 0.0 ? 1.0 : 0.0; }

// ---------------------------------------------------------------------------
// Compile-time assertions
// ---------------------------------------------------------------------------

static_assert(ad::has_absolute_normal_form<^^fn_max_example>());
static_assert(ad::has_absolute_normal_form<^^fn_affine>());
static_assert(ad::has_absolute_normal_form<^^fn_abs_shift>());
static_assert(ad::has_absolute_normal_form<^^fn_min_example>());
static_assert(ad::has_absolute_normal_form<^^fn_max_two_args>());
static_assert(ad::has_absolute_normal_form<^^fn_clamp>());
static_assert(ad::has_absolute_normal_form<^^fn_abs_of_abs>());
static_assert(ad::has_absolute_normal_form<^^fn_abs_const>());
static_assert(ad::has_absolute_normal_form<^^fn_relu>());
static_assert(ad::has_absolute_normal_form<^^fn_kinked_branch>());

static_assert(!ad::has_absolute_normal_form<^^fn_square>());
static_assert(!ad::has_absolute_normal_form<^^fn_exp>());
static_assert(!ad::has_absolute_normal_form<^^fn_ratio>());
static_assert(!ad::has_absolute_normal_form<^^fn_cancelling>());
static_assert(!ad::has_absolute_normal_form<^^fn_step>());

// The worked example, coefficient by coefficient: y = 2 + 2x + |z_0|,
// z_0 = -2 + x.
namespace {
constexpr auto max_plan = ad::absolute_normal_form<^^fn_max_example>::plan;
static_assert(max_plan.switching_variable_count == 1);
static_assert(max_plan.output_form.constant == 2.0);
static_assert(max_plan.output_form.input_coeff[0] == 2.0);
static_assert(max_plan.output_form.switch_coeff[0] == 1.0);
static_assert(max_plan.switching_variables[0].constant == -2.0);
static_assert(max_plan.switching_variables[0].input_coeff[0] == 1.0);

// min is the same rows with the sign of the |z| term flipped.
constexpr auto min_plan = ad::absolute_normal_form<^^fn_min_example>::plan;
static_assert(min_plan.output_form.switch_coeff[0] == -1.0);

// An affine function needs no absolute value at all.
static_assert(ad::anf_result<^^fn_affine>().switching_variable_count == 0);
// |-3.0| is folded into the constant.
static_assert(ad::anf_result<^^fn_abs_const>().switching_variable_count == 0);
// A kink per abs/max/min, and the nested ones chain.
static_assert(ad::anf_result<^^fn_clamp>().switching_variable_count == 2);
static_assert(ad::anf_result<^^fn_abs_of_abs>().switching_variable_count == 2);
static_assert(ad::absolute_normal_form<^^fn_abs_of_abs>::plan
                  .switching_variables[1]
                  .switch_coeff[0] == 1.0);

// relu(x) = x/2 + |x|/2.
constexpr auto relu_plan = ad::absolute_normal_form<^^fn_relu>::plan;
static_assert(relu_plan.switching_variable_count == 1);
static_assert(relu_plan.output_form.input_coeff[0] == 0.5);
static_assert(relu_plan.output_form.switch_coeff[0] == 0.5);

// Where the walk stops on the functions it cannot rewrite.
static_assert(ad::anf_result<^^fn_square>().failing_op == ad::OpKind::Mul);
static_assert(ad::anf_result<^^fn_exp>().failing_op == ad::OpKind::Exp);
static_assert(ad::anf_result<^^fn_ratio>().failing_op == ad::OpKind::Div);
static_assert(ad::anf_result<^^fn_step>().failing_op == ad::OpKind::Select);

// The rewrite is exact, so it holds at compile time too.
static_assert(ad::absolute_normal_form_of<^^fn_max_example>(1.0) == 5.0);
static_assert(ad::absolute_normal_form_of<^^fn_max_example>(3.0) == 9.0);
} // namespace

// ---------------------------------------------------------------------------
// The ANF agrees with the original function, on both sides of every kink
// ---------------------------------------------------------------------------

namespace {

constexpr double samples[] = {-8.0, -3.5, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5,
                              1.0,  1.5,  2.0,  2.5,  3.0,  4.0, 7.25, 12.0};

template <ad::info Fn, typename F> void check_matches(F original) {
  constexpr auto anf = ad::absolute_normal_form<Fn>{};
  for (double x : samples)
    EXPECT_NEAR_ABS(anf(x), original(x), 1e-12);
}

} // namespace

int main() {
  check_matches<^^fn_max_example>(fn_max_example);
  check_matches<^^fn_min_example>(fn_min_example);
  check_matches<^^fn_affine>(fn_affine);
  check_matches<^^fn_abs_shift>(fn_abs_shift);
  check_matches<^^fn_clamp>(fn_clamp);
  check_matches<^^fn_abs_of_abs>(fn_abs_of_abs);
  check_matches<^^fn_abs_const>(fn_abs_const);
  check_matches<^^fn_relu>(fn_relu);
  check_matches<^^fn_kinked_branch>(fn_kinked_branch);

  // Two arguments: max(x, y) = (x + y)/2 + |x - y|/2.
  {
    constexpr auto anf = ad::absolute_normal_form<^^fn_max_two_args>{};
    for (double x : samples)
      for (double y : samples)
        EXPECT_NEAR_ABS(anf(x, y), fn_max_two_args(x, y), 1e-12);
  }

  // absolute_normal_form_of calls the rewritten function without naming it.
  {
    EXPECT_NEAR_ABS(ad::absolute_normal_form_of<^^fn_max_example>(2.0),
                    fn_max_example(2.0), 1e-12);
    EXPECT_NEAR_ABS(ad::absolute_normal_form_of<^^fn_clamp>(-4.0),
                    fn_clamp(-4.0), 1e-12);
    EXPECT_NEAR_ABS(ad::absolute_normal_form_of<^^fn_max_two_args>(1.5, -2.0),
                    fn_max_two_args(1.5, -2.0), 1e-12);
  }

  // The failure report says which node stopped the walk.
  {
    constexpr auto r = ad::anf_result<^^fn_square>();
    EXPECT_FALSE(r.representable);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Mul);
    EXPECT_TRUE(r.failing_node >= 0);
  }

  {
    constexpr auto r = ad::anf_result<^^fn_max_example>();
    EXPECT_TRUE(r.representable);
    EXPECT_EQUAL(r.failing_node, -1);
    EXPECT_EQUAL(r.switching_variable_count, 1u);
  }

  TEST_END;
}
