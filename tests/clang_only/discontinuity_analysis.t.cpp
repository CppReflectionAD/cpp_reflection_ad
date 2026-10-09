#include "discontinuity_analysis.hpp"
#include <test_simple_include.hpp>

#include <algorithm>
#include <cmath>

// Test functions with different numbers of discontinuities

// 0 discontinuities: continuous function
double continuous_linear(double x) { return x + 5.0; }

// 1 discontinuity: single ternary operator
double one_discontinuity(double x) { return (x > 3.0) ? 1.0 : 0.0; }

// 2 discontinuities: two independent ternary operators
double two_discontinuities(double x) {
  return ((x > 2.0) ? 1.0 : 0.0) + ((x > 5.0) ? 1.0 : 0.0);
}

// 3 discontinuities: three independent ternary operators
double three_discontinuities(double x) {
  return ((x > 1.0) ? 1.0 : 0.0) + ((x > 4.0) ? 0.0 : 1.0) +
         ((x > 7.0) ? 2.0 : 0.0);
}

// Multi-argument function: 1 discontinuity w.r.t. first argument, second fixed
// Simple case: target is arg 0, fixed arg 1
double two_arg_compare(double x, double y) { return (x > y) ? 1.0 : 0.0; }

// Digital and call: discontinuity at strike with amplitude 1.0
double digital_and_call_payoff(double spot, double strike) {
  return (spot > strike) ? (1.0 + spot - strike) : 0.0;
}

// Digital call spread: 2 discontinuities with opposite amplitudes
// Returns 1 if strike-1 < spot < strike+1, else 0
double digital_call_spread_payoff(double spot, double strike) {
  return ((spot > strike - 1) ? 1.0 : 0.0) - ((spot > strike + 1) ? 1.0 : 0.0);
}

// Simple negative digital with offset: 1 discontinuity at strike+1
double negative_digital_offset_payoff(double spot, double strike) {
  return (spot > strike + 1) ? -1.0 : 0.0;
}

// Simple positive digital with offset: 1 discontinuity at strike+1
double positive_digital_offset_payoff(double spot, double strike) {
  return (spot > strike + 1) ? 1.0 : 0.0;
}

// Negated digital with offset (negation applied to entire Select)
double negated_select_payoff(double spot, double strike) {
  return -((spot > strike + 1) ? 1.0 : 0.0);
}

// Digital put style payoff: the Heaviside argument has negative derivative in
// spot, so the jump amplitude should be negative as spot increases.
double digital_put_payoff(double spot, double strike) {
  return (spot < strike) ? 1.0 : 0.0;
}

// Other input on the left, target on the right
// strike < spot ? 1.0 : 0.0
double reverse_compare_payoff(double spot, double strike) {
  return (strike < spot) ? 1.0 : 0.0;
}

// #67: `==` / `!=` differ from their surroundings only at a single point, so
// their left and right limits agree and there is no jump to report.
double eq_payoff(double spot, double strike) {
  return (spot == strike) ? 1.0 : 0.0;
}
double ne_payoff(double spot, double strike) {
  return (spot != strike) ? 1.0 : 0.0;
}

// Target on the right-hand side of the comparison, one per operand kind
double const_gt_spot(double spot) { return (100.0 > spot) ? 1.0 : 0.0; }
double strike_plus_one_gt_spot(double spot, double strike) {
  return (strike + 1 > spot) ? 1.0 : 0.0;
}

// Ge with the target on the left (Le is digital_put_le_payoff below)
double digital_call_ge(double spot, double strike) {
  return (spot >= strike) ? 1.0 : 0.0;
}

// #68: the target inside the other operand, under a coefficient, or compared
// against an expression more than one level deep.
double spot_gt_spot_times_k(double spot, double k) {
  return (spot > spot * k) ? 1.0 : 0.0;
}
double spot_gt_strike_minus_spot(double spot, double strike) {
  return (spot > strike - spot) ? 1.0 : 0.0;
}
double spot_gt_two_strike_minus_one(double spot, double strike) {
  return (spot > 2.0 * strike - 1.0) ? 1.0 : 0.0;
}
double two_spot_gt_strike(double spot, double strike) {
  return (2.0 * spot > strike) ? 1.0 : 0.0;
}

// #69: conditions built from comparisons with `!`, `&&`, `||`
double not_digital_call(double spot, double strike) {
  return !(spot > strike) ? 1.0 : 0.0;
}
double range_digital(double spot, double strike) {
  return (spot > strike - 1 && spot < strike + 1) ? 1.0 : 0.0;
}
double outside_range_digital(double spot, double strike) {
  return (spot < strike - 1 || spot > strike + 1) ? 1.0 : 0.0;
}

// #70: the Select's jump scaled on its way to the output
double minus_two_digital_put(double spot, double strike) {
  return -2.0 * ((spot < strike) ? 1.0 : 0.0);
}
double digital_put_times_minus_two(double spot, double strike) {
  return ((spot < strike) ? 1.0 : 0.0) * -2.0;
}
double half_digital_call(double spot, double strike) {
  return 0.5 * ((spot > strike) ? 1.0 : 0.0);
}
double double_negated_digital_call(double spot, double strike) {
  return -(-((spot > strike) ? 1.0 : 0.0));
}

// Jumps are measured on the function, so these come out right too
// A kink, not a jump: continuous at the strike
double call_payoff(double spot, double strike) {
  return (spot > strike) ? spot - strike : 0.0;
}
// Two comparisons flipping at the same point: the jumps add up
double two_steps_at_strike(double spot, double strike) {
  return ((spot > strike) ? 1.0 : 0.0) + ((spot >= strike) ? 1.0 : 0.0);
}
// == at the jump point is held at its off-point outcome on both sides
double step_times_eq(double spot, double strike) {
  return ((spot >= strike) ? 1.0 : 0.0) * ((spot == strike) ? 5.0 : 1.0);
}
// The inner comparison only matters where the outer branch is taken
double nested_digital(double spot, double strike) {
  return (spot > strike) ? ((spot > strike + 10.0) ? 3.0 : 1.0) : 0.0;
}
double unreachable_inner_digital(double spot, double strike) {
  return (spot > strike) ? ((spot < strike - 10.0) ? 3.0 : 1.0) : 0.0;
}

// A comparison in a branch Fn does not take at these arguments is not
// evaluated, so its target-free side cannot fail the build: 1 / k at k = 0
// and log(k) at k < 0 are never computed by Fn
double guarded_reciprocal_digital(double spot, double k) {
  return (k != 0.0) ? ((spot > 1.0 / k) ? 1.0 : 0.0) : 0.0;
}
double guarded_log_digital(double spot, double k) {
  return (k > 0.0) ? ((spot > std::log(k)) ? 1.0 : 0.0) : 0.0;
}
// A guard that varies with the target is evaluated where it can hold for some
// target: at k = 0 `spot > 0.0 && k != 0.0` never does, so 1 / k is skipped,
// while at k = 0.5 it holds above 0 and the inner digital is found
double target_guarded_reciprocal_digital(double spot, double k) {
  return (spot > 0.0 && k != 0.0) ? ((spot > 1.0 / k) ? 1.0 : 0.0) : 0.0;
}
// Unguarded, log(k) at k < 0 is NaN: Fn's comparison is never true
double nan_strike_digital(double spot, double k) {
  return (spot > std::log(k)) ? 1.0 : 0.0;
}

// Roots computed as -g(0)/g' can land an ulp off; each is moved to where Fn's
// own sides are equal, so roots that coincide exactly coincide here too
double third_digital(double spot, double strike) {
  return (spot / 3.0 > strike / 3.0) ? 1.0 : 0.0;
}
double step_plus_seventh_step(double spot, double strike) {
  return ((spot > strike) ? 1.0 : 0.0) +
         ((spot / 7.0 > strike / 7.0) ? 1.0 : 0.0);
}
double step_times_third_eq(double spot, double strike) {
  return ((spot >= strike) ? 1.0 : 0.0) *
         ((spot / 3.0 == strike / 3.0) ? 5.0 : 1.0);
}

// The target need not be the first argument; the other arguments are passed
// in order
double strike_first_digital(double strike, double spot) {
  return (spot > strike) ? 1.0 : 0.0;
}
double three_arg_digital(double lo, double spot, double hi) {
  return (spot > lo && spot < hi) ? 1.0 : 0.0;
}

// A value used directly as a condition tests v != 0: like `!=`, a single
// point, not a jump
double spot_as_condition(double spot) { return spot ? 1.0 : 0.0; }
double not_spot_minus_strike(double spot, double strike) {
  return !(spot - strike) ? 1.0 : 0.0;
}
double spot_minus_strike_and_strike(double spot, double strike) {
  return ((spot - strike) && strike) ? 1.0 : 0.0;
}
// ... but at a shared point it is held at its off-point truth
double step_times_truthy_gap(double spot, double strike) {
  return ((spot >= strike) ? 1.0 : 0.0) * ((spot - strike) ? 1.0 : 5.0);
}
// A condition or comparison of a value that changes only in steps flips
// where the comparison inside it does
double indicator_as_condition(double spot, double strike) {
  const double ind = (spot > strike) ? 1.0 : 0.0;
  return ind ? 3.0 : 1.0;
}
double indicator_compared(double spot, double strike) {
  return (((spot > strike) ? 1.0 : 0.0) > 0.5) ? 2.0 : 0.0;
}

// A pole is not a jump
double reciprocal_above_zero(double spot, double k) {
  return (spot > 0.0) ? k / spot : 0.0;
}

// A crossing scaled by an infinite coefficient cannot be placed: s / k > 1 at
// k = 0 jumps at s = 0, but its slope is inf
double ratio_digital(double spot, double k) {
  return (spot / k > 1.0) ? 1.0 : 0.0;
}
double product_digital(double spot, double k) {
  return (spot * k > 1.0) ? 1.0 : 0.0;
}
// ... nor one whose sides overflow with spot at 0, though Fn computes them
// finitely near the point: at k = 1e300 the first jumps at k, and at
// k = 1e308 the second at about k (where 2 * spot overflows)
double scaled_gap_digital(double spot, double k) {
  return ((spot - k) * 1e10 > 0.0) ? 1.0 : 0.0;
}
double doubled_spot_digital(double spot, double k) {
  return (2.0 * spot - k > k) ? 1.0 : 0.0;
}
// ... while one that is not finite because a value without spot is not
// never flips, and has no point: spot > inf at k = 0 and at k = 1000
double reciprocal_strike_digital(double spot, double k) {
  return (spot > 1.0 / k) ? 1.0 : 0.0;
}
double exp_strike_digital(double spot, double k) {
  return (spot > std::exp(k)) ? 1.0 : 0.0;
}
// A jump whose side passes through an overflow is still bounded: exp(1000)
// is past the largest double, so 1 / (1 + exp(1000)) is within 1e-308 of 0.
double overflowing_rebate(double spot, double k) {
  return (spot > k) ? 1.0 + 1.0 / (1.0 + std::exp(k)) : 0.0;
}
// ... while one whose side has no bound is an error, not a dropped jump: at
// k = 3 the divisor is 4.4e-16, which is rounding alone and may as well be 0
double rounding_rebate(double spot, double k) {
  return (spot > k) ? 1.0 / (k * 0.1 * 10.0 - k) : 0.0;
}

// A crossing behind a guard that cannot hold is never reached, nested or not:
// at k <= 0 the inner `spot > k` is dead, so it is not rooted at k, where
// the other branch's 1 / spot is a pole
double dead_nested_crossing(double spot, double k) {
  return (spot > 1.0) ? ((k > 0.0) ? ((spot > k) ? 1.0 : 0.0) : 0.0)
                      : 1.0 / spot;
}

// ... and so is the else branch of a guard that always holds: at k > 0 the
// condition is always true, so `spot > 0`, whose root 0 is log's pole, is
// never reached
double dead_else_branch(double spot, double k) {
  return (spot > 1.0 || k > 0.0) ? std::log(spot) : (spot > 0.0 ? 1.0 : 0.0);
}

// A `?:` on a target-free condition (a call / put flag) is affine when its
// branches are
double flagged_digital(double spot, double strike, double is_call) {
  return ((is_call > 0.0) ? spot - strike : strike - spot) > 0.0 ? 1.0 : 0.0;
}

// Kinks whose root is not a double: the jump measured at the rounded root is
// rounding, not a discontinuity. In the second, the comparison's sides are
// exactly equal at the snapped root but the branches still are not.
double scaled_call(double spot, double strike) {
  return (spot * 1.1 > strike) ? spot * 1.1 - strike : 0.0;
}
double rounded_strike_call(double spot, double strike) {
  return (spot > strike / 1.1) ? 1.1 * spot - strike : 0.0;
}
// ... while a jump of any size is reported, even next to a steep slope
double tiny_digital(double spot, double strike) {
  return (spot > strike) ? 1e-12 : 0.0;
}
double steep_call_with_rebate(double spot, double strike) {
  return (spot > strike) ? (spot - strike) * 1e6 + 1e-3 : 0.0;
}
// ... however steep, where the root and branches are computed exactly: a
// probe 4 ulps away (8 at 1e16) once read these as rounding at a kink
double steeper_call_with_rebate(double spot, double strike) {
  return (spot > strike) ? (spot - strike) * 1e9 + 1e-5 : 0.0;
}
double call_with_rebate(double spot, double strike) {
  return (spot > strike) ? spot - strike + 1.0 : 0.0;
}
// ... or next to a branch that is NaN on the far side of the point
double sqrt_rebate(double spot, double strike) {
  return (spot > strike) ? std::sqrt(spot - strike) + 1.0 : 0.0;
}
// Kinks whose root is not a double leave rounding at the point, which is not
// reported whatever its order: the branch difference changes sign there (odd
// order) or keeps it (even order, a C^1 join, which a probe 4 ulps away once
// reported as a jump of ~1e-30).
double inexact_kink(double spot, double strike) {
  return (spot > strike / 3.0) ? 3.0 * spot - strike : 0.0;
}
double inexact_scaled_kink(double spot, double strike) {
  return (spot * 1.1 > strike) ? spot - strike / 1.1 : 0.0;
}
double inexact_tangential_kink(double spot, double strike) {
  return (spot > strike / 3.0) ? (3.0 * spot - strike) * (3.0 * spot - strike)
                               : 0.0;
}
double capped_inexact_kink(double spot, double strike) {
  return (spot > strike / 3.0) ? std::min(3.0 * spot - strike, 1.0) : 0.0;
}
// ... and a jump through max and min is measured like any other
double floored_digital(double spot, double strike) {
  return (spot > strike) ? std::max(spot - strike, 1.0)
                         : std::min(spot - strike, 0.0);
}
// max and min of error-bounded values pick the operand Fn does, but the
// other's error bounds the result too
using ad::detail_disc::Ball;
using ad::detail_disc::BallCxMath;
static_assert(BallCxMath::max(Ball{1.0, 0.0}, Ball{0.9, 0.5}).mid == 1.0 &&
              BallCxMath::max(Ball{1.0, 0.0}, Ball{0.9, 0.5}).rad == 0.5);
static_assert(BallCxMath::min(Ball{1.0, 0.25}, Ball{0.9, 0.0}).mid == 0.9 &&
              BallCxMath::min(Ball{1.0, 0.25}, Ball{0.9, 0.0}).rad == 0.25);
// A function's bound covers its argument's whole range, even a radius under
// half an ulp of it (which mid ± rad, rounded to nearest, loses): exp(x) moves
// by e^x · rad, erfc(x) by about 2x · erfc(x) · rad, here well over the
// functions' own error
static_assert(BallCxMath::unary<ad::OpKind::Exp>(Ball{600.0, 5e-14}).rad >=
              5e-14 * cx::exp(600.0));
static_assert(BallCxMath::unary<ad::OpKind::Erfc>(Ball{20.0, 1e-15}).rad >=
              40.0 * 1e-15 * cx::erfc(20.0));
// An overflow is bounded below: past the largest double, with its sign
constexpr Ball exp_1000 = BallCxMath::unary<ad::OpKind::Exp>(Ball{1000.0, 0.0});
constexpr Ball over_exp_1000 =
    BallCxMath::div(Ball{1.0, 0.0}, BallCxMath::add(Ball{1.0, 0.0}, exp_1000));
static_assert(exp_1000.rad < 1e300 && over_exp_1000.mid == 0.0 &&
              over_exp_1000.rad < 1e-307);
static_assert(BallCxMath::mul(Ball{-1e300, 1e290}, Ball{1e10, 0.0}).mid ==
                  -std::numeric_limits<double>::infinity() &&
              BallCxMath::mul(Ball{-1e300, 1e290}, Ball{1e10, 0.0}).rad == 0.0);
static_assert(
    BallCxMath::unary<ad::OpKind::Exp>(BallCxMath::neg(exp_1000)).rad < 1e-307);
// ... unless its sign is not known, or an operand has no bound
static_assert(BallCxMath::mul(exp_1000, Ball{1e-300, 1e-300}).rad ==
              std::numeric_limits<double>::infinity());
static_assert(BallCxMath::add(exp_1000, Ball{-1e308, 1e308}).rad ==
              std::numeric_limits<double>::infinity());
static_assert(BallCxMath::div(Ball{1.0, 0.0}, Ball{4.4e-16, 6.7e-16}).rad ==
              std::numeric_limits<double>::infinity());

// The same crossing written two ways. They meet at strike / 1.1 in exact
// arithmetic but are rooted a few ulps apart, and Fn's own sides can be
// exactly equal on a run of doubles between them. Either way the function
// jumps once, by 1.
double same_crossing_and(double spot, double strike) {
  return (spot * 1.1 > strike && spot / 0.9 > strike / 0.99) ? 1.0 : 0.0;
}
double same_crossing_or(double spot, double strike) {
  return (spot * 1.1 > strike || spot / 0.9 > strike / 0.99) ? 1.0 : 0.0;
}
// ... and summed, it jumps once, by 2
double same_crossing_sum(double spot, double strike) {
  return ((spot * 1.1 > strike) ? 1.0 : 0.0) +
         ((spot / 0.9 > strike / 0.99) ? 1.0 : 0.0);
}

// Functions go through cx_std in both entry points: sin/cos work at compile
// time, and both entry points agree to the bit
double digital_sin_rebate(double spot, double strike) {
  return (spot > strike) ? std::sin(strike) + 1.0 : 0.0;
}
double digital_exp_payoff(double spot, double strike) {
  return (spot > strike) ? std::exp(spot / 10.0) : 0.0;
}

// Jump of payoff(spot, strike) as spot crosses `point` upward.
double measure_jump_on_function(double (*payoff)(double, double), double point,
                                double strike, double epsilon = 1e-8) {
  return payoff(point + epsilon, strike) - payoff(point - epsilon, strike);
}

// Constant on the left, target on the right
// 99.0 < spot ? 1.0 : 0.0
double const_compare_payoff(double spot) { return (99.0 < spot) ? 1.0 : 0.0; }

// Expression on the left, target on the right
// strike + 1 < spot ? 1.0 : 0.0
double offset_compare_payoff(double spot, double strike) {
  return (strike + 1 < spot) ? 1.0 : 0.0;
}

// Expression on the left with the constant first, target on the right
// 1 + strike < spot ? 1.0 : 0.0
double offset_compare_sum_payoff(double spot, double strike) {
  return (1 + strike < spot) ? 1.0 : 0.0;
}

// A digital put with <=, target on the left
// spot <= strike ? 1.0 : 0.0
double digital_put_le_payoff(double spot, double strike) {
  return (spot <= strike) ? 1.0 : 0.0;
}

// A digital put with >=, target on the right
// strike >= spot ? 1.0 : 0.0
double digital_put_ge_payoff(double spot, double strike) {
  return (strike >= spot) ? 1.0 : 0.0;
}

int main() {
  // Test 0 discontinuities
  constexpr auto disc0 = ad::get_discontinuity_points<^^continuous_linear, 0>();
  // Should be empty (no discontinuities)
  EXPECT_TRUE(disc0.empty());
  EXPECT_EQUAL(disc0.size(), 0);

  // Test 1 discontinuity
  constexpr auto disc1 = ad::get_discontinuity_points<^^one_discontinuity, 0>();
  EXPECT_FALSE(disc1.empty());
  EXPECT_EQUAL(disc1.size(), 1);
  EXPECT_EQUAL(disc1[0], 3.0);

  // Test 2 discontinuities
  constexpr auto disc2 =
      ad::get_discontinuity_points<^^two_discontinuities, 0>();
  EXPECT_FALSE(disc2.empty());
  EXPECT_EQUAL(disc2.size(), 2);
  EXPECT_EQUAL(disc2[0], 2.0);
  EXPECT_EQUAL(disc2[1], 5.0);

  // Test 3 discontinuities
  constexpr auto disc3 =
      ad::get_discontinuity_points<^^three_discontinuities, 0>();
  EXPECT_FALSE(disc3.empty());
  EXPECT_EQUAL(disc3.size(), 3);
  EXPECT_EQUAL(disc3[0], 1.0);
  EXPECT_EQUAL(disc3[1], 4.0);
  EXPECT_EQUAL(disc3[2], 7.0);

  // Test multi-argument function
  // two_arg_compare(x, y) where x is target (arg 0) and y is fixed at 5.0
  // Discontinuity at x = y = 5.0
  constexpr auto disc_2arg =
      ad::get_discontinuity_points<^^two_arg_compare, 0>(5.0);
  EXPECT_FALSE(disc_2arg.empty());
  EXPECT_EQUAL(disc_2arg.size(), 1);
  EXPECT_EQUAL(disc_2arg[0], 5.0);

  // Test get_discontinuity_points_and_amplitudes
  // three_discontinuities: ((x > 1.0) ? 1.0 : 0.0) + ((x > 4.0) ? 0.0 : 1.0) +
  // ((x > 7.0) ? 2.0 : 0.0)
  // Expected:
  //   x=1.0: amp=1.0 (from first ternary)
  //   x=4.0: amp=-1.0 (from second ternary, inverted)
  //   x=7.0: amp=2.0 (from third ternary)
  constexpr auto disc_amp =
      ad::get_discontinuity_points_and_amplitudes<^^three_discontinuities, 0>();
  EXPECT_FALSE(disc_amp.empty());
  EXPECT_EQUAL(disc_amp.size(), 3);
  // First point
  EXPECT_EQUAL(disc_amp.point(0), 1.0);
  EXPECT_EQUAL(disc_amp.amplitude(0), 1.0);
  // Second point
  EXPECT_EQUAL(disc_amp.point(1), 4.0);
  EXPECT_EQUAL(disc_amp.amplitude(1), -1.0);
  // Third point
  EXPECT_EQUAL(disc_amp.point(2), 7.0);
  EXPECT_EQUAL(disc_amp.amplitude(2), 2.0);

  // Test with one_discontinuity and amplitudes
  constexpr auto disc_1_amp =
      ad::get_discontinuity_points_and_amplitudes<^^one_discontinuity, 0>();
  EXPECT_FALSE(disc_1_amp.empty());
  EXPECT_EQUAL(disc_1_amp.size(), 1);
  EXPECT_EQUAL(disc_1_amp.point(0), 3.0);
  EXPECT_EQUAL(disc_1_amp.amplitude(0), 1.0);

  // Test digital_and_call_payoff with amplitudes
  // (spot > strike) ? (1.0 + spot - strike) : 0.0
  // At strike = 100.0:
  //   True branch at 100.0: 1.0 + 100 - 100 = 1.0
  //   False branch: 0.0
  //   Amplitude: 1.0 - 0.0 = 1.0
  constexpr auto disc_dac =
      ad::get_discontinuity_points_and_amplitudes<^^digital_and_call_payoff, 0>(
          100.0);
  EXPECT_FALSE(disc_dac.empty());
  EXPECT_EQUAL(disc_dac.size(), 1);
  EXPECT_EQUAL(disc_dac.point(0), 100.0);
  EXPECT_EQUAL(disc_dac.amplitude(0), 1.0);

  // Test digital_call_spread_payoff with amplitudes
  // ((spot > strike - 1) ? 1.0 : 0.0) - ((spot > strike + 1) ? 1.0 : 0.0)
  // At strike = 100.0:
  //   First discontinuity at spot = 99.0 (strike - 1): amplitude = 1.0
  //   Second discontinuity at spot = 101.0 (strike + 1): amplitude = -1.0
  constexpr auto disc_spread =
      ad::get_discontinuity_points_and_amplitudes<^^digital_call_spread_payoff,
                                                  0>(100.0);
  EXPECT_FALSE(disc_spread.empty());
  EXPECT_EQUAL(disc_spread.size(), 2);
  EXPECT_EQUAL(disc_spread.point(0), 99.0);
  EXPECT_EQUAL(disc_spread.amplitude(0), 1.0);
  EXPECT_EQUAL(disc_spread.point(1), 101.0);
  EXPECT_EQUAL(disc_spread.amplitude(1), -1.0);

  // Test negative_digital_offset_payoff with amplitudes
  // (spot > strike + 1) ? -1.0 : 0.0
  // At strike = 100.0:
  //   True branch at 101.0: -1.0
  //   False branch: 0.0
  //   Amplitude: -1.0 - 0.0 = -1.0
  constexpr auto disc_neg = ad::get_discontinuity_points_and_amplitudes<
      ^^negative_digital_offset_payoff, 0>(100.0);
  EXPECT_FALSE(disc_neg.empty());
  EXPECT_EQUAL(disc_neg.size(), 1);
  EXPECT_EQUAL(disc_neg.point(0), 101.0);
  EXPECT_EQUAL(disc_neg.amplitude(0), -1.0);

  // Test positive_digital_offset_payoff with amplitudes
  // (spot > strike + 1) ? 1.0 : 0.0
  // At strike = 100.0:
  //   True branch at 101.0: 1.0
  //   False branch: 0.0
  //   Amplitude: 1.0 - 0.0 = 1.0
  constexpr auto disc_pos = ad::get_discontinuity_points_and_amplitudes<
      ^^positive_digital_offset_payoff, 0>(100.0);
  EXPECT_FALSE(disc_pos.empty());
  EXPECT_EQUAL(disc_pos.size(), 1);
  EXPECT_EQUAL(disc_pos.point(0), 101.0);
  EXPECT_EQUAL(disc_pos.amplitude(0), 1.0);

  // Test negated_select_payoff with amplitudes
  // -((spot > strike + 1) ? 1.0 : 0.0)
  // At strike = 100.0:
  //   Select: true_branch = 1.0, false_branch = 0.0, amplitude = 1.0
  //   Negation applied: amplitude = -1.0
  constexpr auto disc_neg_sel =
      ad::get_discontinuity_points_and_amplitudes<^^negated_select_payoff, 0>(
          100.0);
  EXPECT_FALSE(disc_neg_sel.empty());
  EXPECT_EQUAL(disc_neg_sel.size(), 1);
  EXPECT_EQUAL(disc_neg_sel.point(0), 101.0);
  EXPECT_EQUAL(disc_neg_sel.amplitude(0), -1.0);

  // Test digital_put_payoff with amplitudes
  // (spot < strike) ? 1.0 : 0.0
  // At strike = 100.0, the jump as spot increases is 0.0 - 1.0 = -1.0.
  constexpr auto disc_put =
      ad::get_discontinuity_points_and_amplitudes<^^digital_put_payoff, 0>(
          100.0);
  EXPECT_FALSE(disc_put.empty());
  EXPECT_EQUAL(disc_put.size(), 1);
  EXPECT_EQUAL(disc_put.point(0), 100.0);
  EXPECT_EQUAL(disc_put.amplitude(0), -1.0);

  // Test digital call with amplitudes using two_arg_compare
  // (x > y) ? 1.0 : 0.0, with y fixed at 100.0
  // At y = 100.0, the jump as x increases is 1.0 - 0.0 = +1.0.
  constexpr auto disc_call =
      ad::get_discontinuity_points_and_amplitudes<^^two_arg_compare, 0>(100.0);
  EXPECT_FALSE(disc_call.empty());
  EXPECT_EQUAL(disc_call.size(), 1);
  EXPECT_EQUAL(disc_call.point(0), 100.0);
  EXPECT_EQUAL(disc_call.amplitude(0), 1.0);

  // Test runtime version with runtime arguments
  // get_discontinuity_points_and_amplitudes_rt exercises the compile-time DAG
  // analysis with runtime parameter values
  double runtime_strike = 100.0;
  auto disc_rt =
      ad::get_discontinuity_points_and_amplitudes_rt<^^digital_and_call_payoff,
                                                     0>(runtime_strike);
  EXPECT_FALSE(disc_rt.empty());
  EXPECT_EQUAL(disc_rt.size(), 1);
  EXPECT_EQUAL(disc_rt.point(0), 100.0);
  EXPECT_EQUAL(disc_rt.amplitude(0), 1.0);

  // Constant on the left: const < target
  // (99.0 < spot) ? 1.0 : 0.0
  // Discontinuity at spot = 99.0 with amplitude +1.0
  constexpr auto disc_const_lhs =
      ad::get_discontinuity_points_and_amplitudes<^^const_compare_payoff, 0>();
  EXPECT_FALSE(disc_const_lhs.empty());
  EXPECT_EQUAL(disc_const_lhs.size(), 1);
  EXPECT_EQUAL(disc_const_lhs.point(0), 99.0);
  EXPECT_EQUAL(disc_const_lhs.amplitude(0), 1.0);

  // Other input on the left: other_input < target
  // (strike < spot) ? 1.0 : 0.0
  // Discontinuity at spot = strike with amplitude +1.0
  constexpr auto disc_strike_lhs =
      ad::get_discontinuity_points_and_amplitudes<^^reverse_compare_payoff, 0>(
          100.0);
  EXPECT_FALSE(disc_strike_lhs.empty());
  EXPECT_EQUAL(disc_strike_lhs.size(), 1);
  EXPECT_EQUAL(disc_strike_lhs.point(0), 100.0);
  EXPECT_EQUAL(disc_strike_lhs.amplitude(0), 1.0);

  // Expression on the left: expression < target
  // (strike + 1 < spot) ? 1.0 : 0.0
  // Discontinuity at spot = strike + 1 with amplitude +1.0
  constexpr auto disc_offset_lhs =
      ad::get_discontinuity_points_and_amplitudes<^^offset_compare_payoff, 0>(
          100.0);
  EXPECT_FALSE(disc_offset_lhs.empty());
  EXPECT_EQUAL(disc_offset_lhs.size(), 1);
  EXPECT_EQUAL(disc_offset_lhs.point(0), 101.0);
  EXPECT_EQUAL(disc_offset_lhs.amplitude(0), 1.0);

  // Expression on the left, constant first: expression < target
  // (1 + strike < spot) ? 1.0 : 0.0
  // Discontinuity at spot = 1 + strike with amplitude +1.0
  constexpr auto disc_offset_sum_lhs =
      ad::get_discontinuity_points_and_amplitudes<^^offset_compare_sum_payoff,
                                                  0>(100.0);
  EXPECT_FALSE(disc_offset_sum_lhs.empty());
  EXPECT_EQUAL(disc_offset_sum_lhs.size(), 1);
  EXPECT_EQUAL(disc_offset_sum_lhs.point(0), 101.0);
  EXPECT_EQUAL(disc_offset_sum_lhs.amplitude(0), 1.0);

  // Le: a digital put
  // (spot <= strike) ? 1.0 : 0.0
  // At strike = 100.0, jump as spot increases is -1.0 (1.0 -> 0.0)
  constexpr auto disc_le =
      ad::get_discontinuity_points_and_amplitudes<^^digital_put_le_payoff, 0>(
          100.0);
  EXPECT_FALSE(disc_le.empty());
  EXPECT_EQUAL(disc_le.size(), 1);
  EXPECT_EQUAL(disc_le.point(0), 100.0);
  EXPECT_EQUAL(disc_le.amplitude(0), -1.0);

  // Ge with the target on the right: a digital put
  // (strike >= spot) ? 1.0 : 0.0
  // At strike = 100.0, this is true on left (spot <= 100), false on right
  // Jump as spot increases is -1.0 (1.0 -> 0.0)
  constexpr auto disc_ge =
      ad::get_discontinuity_points_and_amplitudes<^^digital_put_ge_payoff, 0>(
          100.0);
  EXPECT_FALSE(disc_ge.empty());
  EXPECT_EQUAL(disc_ge.size(), 1);
  EXPECT_EQUAL(disc_ge.point(0), 100.0);
  EXPECT_EQUAL(disc_ge.amplitude(0), -1.0);

  // Runtime version, other input on the left
  double runtime_strike_lhs = 100.0;
  auto disc_strike_lhs_rt =
      ad::get_discontinuity_points_and_amplitudes_rt<^^reverse_compare_payoff,
                                                     0>(runtime_strike_lhs);
  EXPECT_FALSE(disc_strike_lhs_rt.empty());
  EXPECT_EQUAL(disc_strike_lhs_rt.size(), 1);
  EXPECT_EQUAL(disc_strike_lhs_rt.point(0), 100.0);
  EXPECT_EQUAL(disc_strike_lhs_rt.amplitude(0), 1.0);

  // #67: equality comparisons are not discontinuities
  EXPECT_EQUAL((ad::get_discontinuity_points<^^eq_payoff, 0>(100.0).size()), 0);
  EXPECT_EQUAL((ad::get_discontinuity_points<^^ne_payoff, 0>(100.0).size()), 0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^eq_payoff, 0>(100.0)
           .size()),
      0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^ne_payoff, 0>(100.0)
           .size()),
      0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^eq_payoff, 0>(100.0)
           .size()),
      0);

  // Constant on the left, target on the right
  constexpr auto disc_c2 =
      ad::get_discontinuity_points_and_amplitudes<^^const_gt_spot, 0>();
  EXPECT_EQUAL(disc_c2.size(), 1);
  EXPECT_EQUAL(disc_c2.point(0), 100.0);
  EXPECT_EQUAL(disc_c2.amplitude(0), -1.0);
  EXPECT_EQUAL(const_gt_spot(100.0 + 1e-8) - const_gt_spot(100.0 - 1e-8),
               disc_c2.amplitude(0));

  // The cases above, measured on the function itself
  EXPECT_EQUAL(const_compare_payoff(99.0 + 1e-8) -
                   const_compare_payoff(99.0 - 1e-8),
               disc_const_lhs.amplitude(0));
  EXPECT_EQUAL(measure_jump_on_function(reverse_compare_payoff, 100.0, 100.0),
               disc_strike_lhs.amplitude(0));
  EXPECT_EQUAL(measure_jump_on_function(offset_compare_payoff, 101.0, 100.0),
               disc_offset_lhs.amplitude(0));
  EXPECT_EQUAL(
      measure_jump_on_function(offset_compare_sum_payoff, 101.0, 100.0),
      disc_offset_sum_lhs.amplitude(0));
  EXPECT_EQUAL(measure_jump_on_function(digital_put_le_payoff, 100.0, 100.0),
               disc_le.amplitude(0));
  EXPECT_EQUAL(measure_jump_on_function(digital_put_ge_payoff, 100.0, 100.0),
               disc_ge.amplitude(0));

  // Binary op on the left, target on the right
  constexpr auto disc_c6 =
      ad::get_discontinuity_points_and_amplitudes<^^strike_plus_one_gt_spot, 0>(
          100.0);
  EXPECT_EQUAL(disc_c6.size(), 1);
  EXPECT_EQUAL(disc_c6.point(0), 101.0);
  EXPECT_EQUAL(disc_c6.amplitude(0), -1.0);
  EXPECT_EQUAL(measure_jump_on_function(strike_plus_one_gt_spot, 101.0, 100.0),
               disc_c6.amplitude(0));

  // Ge with the target on the left
  constexpr auto disc_call_ge =
      ad::get_discontinuity_points_and_amplitudes<^^digital_call_ge, 0>(100.0);
  EXPECT_EQUAL(disc_call_ge.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(digital_call_ge, 100.0, 100.0),
               disc_call_ge.amplitude(0));

  // The runtime entry point shares the analysis
  const auto disc_put_rt =
      ad::get_discontinuity_points_and_amplitudes_rt<^^digital_put_payoff, 0>(
          100.0);
  EXPECT_EQUAL(disc_put_rt.size(), 1);
  EXPECT_EQUAL(disc_put_rt.point(0), 100.0);
  EXPECT_EQUAL(disc_put_rt.amplitude(0), -1.0);

  // #68: the crossing is the root of lhs - rhs, whatever the shape.
  // s > s * k with k = 2: true for s < 0, so the jump at 0 is -1.
  constexpr auto disc_sk =
      ad::get_discontinuity_points_and_amplitudes<^^spot_gt_spot_times_k, 0>(
          2.0);
  EXPECT_EQUAL(disc_sk.size(), 1);
  EXPECT_EQUAL(disc_sk.point(0), 0.0);
  EXPECT_EQUAL(disc_sk.amplitude(0), -1.0);
  EXPECT_EQUAL(measure_jump_on_function(spot_gt_spot_times_k, 0.0, 2.0),
               disc_sk.amplitude(0));
  // s > k - s: crosses at k / 2
  constexpr auto disc_kms =
      ad::get_discontinuity_points_and_amplitudes<^^spot_gt_strike_minus_spot,
                                                  0>(100.0);
  EXPECT_EQUAL(disc_kms.size(), 1);
  EXPECT_EQUAL(disc_kms.point(0), 50.0);
  EXPECT_EQUAL(disc_kms.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(spot_gt_strike_minus_spot, 50.0, 100.0),
               disc_kms.amplitude(0));
  // s > 2k - 1: a two-level operand, crosses at 199
  constexpr auto disc_2k = ad::get_discontinuity_points_and_amplitudes<
      ^^spot_gt_two_strike_minus_one, 0>(100.0);
  EXPECT_EQUAL(disc_2k.size(), 1);
  EXPECT_EQUAL(disc_2k.point(0), 199.0);
  EXPECT_EQUAL(disc_2k.amplitude(0), 1.0);
  EXPECT_EQUAL(
      measure_jump_on_function(spot_gt_two_strike_minus_one, 199.0, 100.0),
      disc_2k.amplitude(0));
  // 2s > k: the target under a coefficient, crosses at 50
  constexpr auto disc_2s =
      ad::get_discontinuity_points_and_amplitudes<^^two_spot_gt_strike, 0>(
          100.0);
  EXPECT_EQUAL(disc_2s.size(), 1);
  EXPECT_EQUAL(disc_2s.point(0), 50.0);
  EXPECT_EQUAL(disc_2s.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(two_spot_gt_strike, 50.0, 100.0),
               disc_2s.amplitude(0));
  EXPECT_EQUAL(
      (ad::get_discontinuity_points<^^two_spot_gt_strike, 0>(100.0)[0]), 50.0);

  // #69: `!(s > k)` is a digital put
  constexpr auto disc_not =
      ad::get_discontinuity_points_and_amplitudes<^^not_digital_call, 0>(100.0);
  EXPECT_EQUAL(disc_not.size(), 1);
  EXPECT_EQUAL(disc_not.point(0), 100.0);
  EXPECT_EQUAL(disc_not.amplitude(0), -1.0);
  EXPECT_EQUAL(measure_jump_on_function(not_digital_call, 100.0, 100.0),
               disc_not.amplitude(0));
  // && : a range digital steps up at k - 1 and down at k + 1
  constexpr auto disc_and =
      ad::get_discontinuity_points_and_amplitudes<^^range_digital, 0>(100.0);
  EXPECT_EQUAL(disc_and.size(), 2);
  EXPECT_EQUAL(disc_and.point(0), 99.0);
  EXPECT_EQUAL(disc_and.amplitude(0), 1.0);
  EXPECT_EQUAL(disc_and.point(1), 101.0);
  EXPECT_EQUAL(disc_and.amplitude(1), -1.0);
  EXPECT_EQUAL(measure_jump_on_function(range_digital, 99.0, 100.0),
               disc_and.amplitude(0));
  EXPECT_EQUAL(measure_jump_on_function(range_digital, 101.0, 100.0),
               disc_and.amplitude(1));
  // || : its complement
  const auto disc_or =
      ad::get_discontinuity_points_and_amplitudes_rt<^^outside_range_digital,
                                                     0>(100.0);
  EXPECT_EQUAL(disc_or.size(), 2);
  EXPECT_EQUAL(disc_or.point(0), 99.0);
  EXPECT_EQUAL(disc_or.amplitude(0), -1.0);
  EXPECT_EQUAL(disc_or.point(1), 101.0);
  EXPECT_EQUAL(disc_or.amplitude(1), 1.0);

  // #70: the jump is scaled by whatever sits between the Select and the output
  constexpr auto disc_m2put =
      ad::get_discontinuity_points_and_amplitudes<^^minus_two_digital_put, 0>(
          100.0);
  EXPECT_EQUAL(disc_m2put.size(), 1);
  EXPECT_EQUAL(disc_m2put.amplitude(0), 2.0);
  EXPECT_EQUAL(measure_jump_on_function(minus_two_digital_put, 100.0, 100.0),
               disc_m2put.amplitude(0));
  constexpr auto disc_putm2 =
      ad::get_discontinuity_points_and_amplitudes<^^digital_put_times_minus_two,
                                                  0>(100.0);
  EXPECT_EQUAL(disc_putm2.amplitude(0), 2.0);
  constexpr auto disc_half =
      ad::get_discontinuity_points_and_amplitudes<^^half_digital_call, 0>(
          100.0);
  EXPECT_EQUAL(disc_half.amplitude(0), 0.5);
  constexpr auto disc_negneg =
      ad::get_discontinuity_points_and_amplitudes<^^double_negated_digital_call,
                                                  0>(100.0);
  EXPECT_EQUAL(disc_negneg.amplitude(0), 1.0);

  // A kink has no jump, so it is not a discontinuity point
  EXPECT_EQUAL((ad::get_discontinuity_points<^^call_payoff, 0>(100.0).size()),
               0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^call_payoff, 0>(100.0)
           .size()),
      0);
  // Jumps at a shared point add up
  constexpr auto disc_two =
      ad::get_discontinuity_points_and_amplitudes<^^two_steps_at_strike, 0>(
          100.0);
  EXPECT_EQUAL(disc_two.size(), 1);
  EXPECT_EQUAL(disc_two.amplitude(0), 2.0);
  EXPECT_EQUAL(measure_jump_on_function(two_steps_at_strike, 100.0, 100.0),
               disc_two.amplitude(0));
  // Left limit 0, right limit 1 * 1: the 5 at the point itself is not a jump
  constexpr auto disc_eq =
      ad::get_discontinuity_points_and_amplitudes<^^step_times_eq, 0>(100.0);
  EXPECT_EQUAL(disc_eq.size(), 1);
  EXPECT_EQUAL(disc_eq.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(step_times_eq, 100.0, 100.0),
               disc_eq.amplitude(0));
  // Nested selects: 0 -> 1 at k, 1 -> 3 at k + 10
  constexpr auto disc_nested =
      ad::get_discontinuity_points_and_amplitudes<^^nested_digital, 0>(100.0);
  EXPECT_EQUAL(disc_nested.size(), 2);
  EXPECT_EQUAL(disc_nested.point(0), 100.0);
  EXPECT_EQUAL(disc_nested.amplitude(0), 1.0);
  EXPECT_EQUAL(disc_nested.point(1), 110.0);
  EXPECT_EQUAL(disc_nested.amplitude(1), 2.0);
  // The inner comparison flips at k - 10, inside the untaken branch
  constexpr auto disc_unreach =
      ad::get_discontinuity_points_and_amplitudes<^^unreachable_inner_digital,
                                                  0>(100.0);
  EXPECT_EQUAL(disc_unreach.size(), 1);
  EXPECT_EQUAL(disc_unreach.point(0), 100.0);
  EXPECT_EQUAL(disc_unreach.amplitude(0), 1.0);

  // Guarded target-free operands are not evaluated where Fn skips them
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^guarded_reciprocal_digital,
                                                   0>(0.0)
           .size()),
      0);
  constexpr auto disc_recip =
      ad::get_discontinuity_points_and_amplitudes<^^guarded_reciprocal_digital,
                                                  0>(0.5);
  EXPECT_EQUAL(disc_recip.size(), 1);
  EXPECT_EQUAL(disc_recip.point(0), 2.0);
  EXPECT_EQUAL(disc_recip.amplitude(0), 1.0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^guarded_log_digital, 0>(
           -1.0)
           .size()),
      0);
  constexpr auto disc_glog =
      ad::get_discontinuity_points_and_amplitudes<^^guarded_log_digital, 0>(
          1.0);
  EXPECT_EQUAL(disc_glog.size(), 1);
  EXPECT_EQUAL(disc_glog.point(0), 0.0);
  EXPECT_EQUAL(disc_glog.amplitude(0), 1.0);
  EXPECT_EQUAL((ad::get_discontinuity_points_and_amplitudes<
                    ^^target_guarded_reciprocal_digital, 0>(0.0)
                    .size()),
               0);
  constexpr auto disc_tgrecip = ad::get_discontinuity_points_and_amplitudes<
      ^^target_guarded_reciprocal_digital, 0>(0.5);
  EXPECT_EQUAL(disc_tgrecip.size(), 1);
  EXPECT_EQUAL(disc_tgrecip.point(0), 2.0);
  EXPECT_EQUAL(disc_tgrecip.amplitude(0), 1.0);
  // A dead nested crossing is not rooted (its root, 0, is a pole of the live
  // branch); a live one is
  constexpr auto disc_dead =
      ad::get_discontinuity_points_and_amplitudes<^^dead_nested_crossing, 0>(
          0.0);
  EXPECT_EQUAL(disc_dead.size(), 1);
  EXPECT_EQUAL(disc_dead.point(0), 1.0);
  EXPECT_EQUAL(disc_dead.amplitude(0), -1.0);
  constexpr auto disc_live =
      ad::get_discontinuity_points_and_amplitudes<^^dead_nested_crossing, 0>(
          2.0);
  EXPECT_EQUAL(disc_live.size(), 2);
  EXPECT_EQUAL(disc_live.point(0), 1.0);
  EXPECT_EQUAL(disc_live.amplitude(0), -1.0);
  EXPECT_EQUAL(disc_live.point(1), 2.0);
  EXPECT_EQUAL(disc_live.amplitude(1), 1.0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^dead_nested_crossing,
                                                      0>(0.0)
           .size()),
      1);

  // A dead else branch is not rooted either; a live one is
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^dead_else_branch, 0>(1.0)
           .size()),
      0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^dead_else_branch, 0>(
           1.0)
           .size()),
      0);
  constexpr auto disc_live_else =
      ad::get_discontinuity_points_and_amplitudes<^^dead_else_branch, 0>(-1.0);
  EXPECT_EQUAL(disc_live_else.size(), 2);
  EXPECT_EQUAL(disc_live_else.point(0), 0.0);
  EXPECT_EQUAL(disc_live_else.amplitude(0), 1.0);
  EXPECT_EQUAL(disc_live_else.point(1), 1.0);
  EXPECT_EQUAL(disc_live_else.amplitude(1), -1.0);

  // A target-free flag picks the side: a digital call, or a digital put
  constexpr auto disc_flag_call =
      ad::get_discontinuity_points_and_amplitudes<^^flagged_digital, 0>(100.0,
                                                                        1.0);
  EXPECT_EQUAL(disc_flag_call.size(), 1);
  EXPECT_EQUAL(disc_flag_call.point(0), 100.0);
  EXPECT_EQUAL(disc_flag_call.amplitude(0), 1.0);
  constexpr auto disc_flag_put =
      ad::get_discontinuity_points_and_amplitudes<^^flagged_digital, 0>(100.0,
                                                                        0.0);
  EXPECT_EQUAL(disc_flag_put.size(), 1);
  EXPECT_EQUAL(disc_flag_put.point(0), 100.0);
  EXPECT_EQUAL(disc_flag_put.amplitude(0), -1.0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^flagged_digital, 0>(
           100.0, 0.0)
           .amplitude(0)),
      -1.0);

  // Kinks whose root is not a double are not discontinuities
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^scaled_call, 0>(99.0)
           .size()),
      0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^rounded_strike_call, 0>(
           99.0)
           .size()),
      0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^scaled_call, 0>(99.0)
           .size()),
      0);
  // ... but a jump is, however small
  constexpr auto disc_tiny =
      ad::get_discontinuity_points_and_amplitudes<^^tiny_digital, 0>(1e6);
  EXPECT_EQUAL(disc_tiny.size(), 1);
  EXPECT_EQUAL(disc_tiny.point(0), 1e6);
  EXPECT_EQUAL(disc_tiny.amplitude(0), 1e-12);
  constexpr auto disc_steep =
      ad::get_discontinuity_points_and_amplitudes<^^steep_call_with_rebate, 0>(
          100.0);
  EXPECT_EQUAL(disc_steep.size(), 1);
  EXPECT_EQUAL(disc_steep.point(0), 100.0);
  EXPECT_EQUAL(disc_steep.amplitude(0), 1e-3);
  constexpr auto disc_steeper =
      ad::get_discontinuity_points_and_amplitudes<^^steeper_call_with_rebate,
                                                  0>(100.0);
  EXPECT_EQUAL(disc_steeper.size(), 1);
  EXPECT_EQUAL(disc_steeper.point(0), 100.0);
  EXPECT_EQUAL(disc_steeper.amplitude(0), 1e-5);
  constexpr auto disc_far_rebate =
      ad::get_discontinuity_points_and_amplitudes<^^call_with_rebate, 0>(1e16);
  EXPECT_EQUAL(disc_far_rebate.size(), 1);
  EXPECT_EQUAL(disc_far_rebate.point(0), 1e16);
  EXPECT_EQUAL(disc_far_rebate.amplitude(0), 1.0);
  {
    // Kinks with inexact roots, odd or even order, report nothing anywhere.
    int odd_points = 0, scaled_points = 0, tangential_points = 0,
        capped_points = 0;
    for (int i = 1; i <= 400; ++i) {
      const double strike = 0.37 * i;
      odd_points +=
          ad::get_discontinuity_points_and_amplitudes_rt<^^inexact_kink, 0>(
              strike)
              .size();
      scaled_points +=
          ad::get_discontinuity_points_and_amplitudes_rt<^^inexact_scaled_kink,
                                                         0>(strike)
              .size();
      tangential_points += ad::get_discontinuity_points_and_amplitudes_rt<
                               ^^inexact_tangential_kink, 0>(strike)
                               .size();
      capped_points +=
          ad::get_discontinuity_points_and_amplitudes_rt<^^capped_inexact_kink,
                                                         0>(strike)
              .size();
    }
    EXPECT_EQUAL(odd_points, 0);
    EXPECT_EQUAL(scaled_points, 0);
    EXPECT_EQUAL(tangential_points, 0);
    EXPECT_EQUAL(capped_points, 0);
    // ... at compile time too: 7.03 / 3 is not a double
    EXPECT_TRUE(
        (ad::get_discontinuity_points_and_amplitudes<^^inexact_tangential_kink,
                                                     0>(7.03)
             .empty()));
  }
  constexpr auto disc_floored =
      ad::get_discontinuity_points_and_amplitudes<^^floored_digital, 0>(100.0);
  EXPECT_EQUAL(disc_floored.size(), 1);
  EXPECT_EQUAL(disc_floored.point(0), 100.0);
  EXPECT_EQUAL(disc_floored.amplitude(0), 1.0);
  constexpr auto disc_sqrt =
      ad::get_discontinuity_points_and_amplitudes<^^sqrt_rebate, 0>(100.0);
  EXPECT_EQUAL(disc_sqrt.size(), 1);
  EXPECT_EQUAL(disc_sqrt.point(0), 100.0);
  EXPECT_EQUAL(disc_sqrt.amplitude(0), 1.0);
  {
    const auto disc_sqrt_rt =
        ad::get_discontinuity_points_and_amplitudes_rt<^^sqrt_rebate, 0>(100.0);
    EXPECT_EQUAL(disc_sqrt_rt.size(), 1);
    EXPECT_EQUAL(disc_sqrt_rt.point(0), 100.0);
    EXPECT_EQUAL(disc_sqrt_rt.amplitude(0), 1.0);
  }
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^nan_strike_digital, 0>(
           -1.0)
           .size()),
      0);
  EXPECT_EQUAL((ad::get_discontinuity_points_and_amplitudes_rt<
                    ^^guarded_reciprocal_digital, 0>(0.0)
                    .size()),
               0);

  // Coincident roots are measured together
  constexpr auto disc_third =
      ad::get_discontinuity_points_and_amplitudes<^^third_digital, 0>(100.0);
  EXPECT_EQUAL(disc_third.size(), 1);
  EXPECT_EQUAL(disc_third.point(0), 100.0);
  EXPECT_EQUAL(disc_third.amplitude(0), 1.0);
  constexpr auto disc_seventh =
      ad::get_discontinuity_points_and_amplitudes<^^step_plus_seventh_step, 0>(
          100.0);
  EXPECT_EQUAL(disc_seventh.size(), 1);
  EXPECT_EQUAL(disc_seventh.point(0), 100.0);
  EXPECT_EQUAL(disc_seventh.amplitude(0), 2.0);
  EXPECT_EQUAL(measure_jump_on_function(step_plus_seventh_step, 100.0, 100.0),
               disc_seventh.amplitude(0));
  constexpr auto disc_third_eq =
      ad::get_discontinuity_points_and_amplitudes<^^step_times_third_eq, 0>(
          100.0);
  EXPECT_EQUAL(disc_third_eq.size(), 1);
  EXPECT_EQUAL(disc_third_eq.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(step_times_third_eq, 100.0, 100.0),
               disc_third_eq.amplitude(0));

  // The same crossing written two ways jumps once, by 1, wherever its two
  // roots land: at 69.19 Fn's sides are both exactly equal on a run of
  // doubles, and each root is on the other's run
  constexpr auto disc_same_and =
      ad::get_discontinuity_points_and_amplitudes<^^same_crossing_and, 0>(
          69.19);
  EXPECT_EQUAL(disc_same_and.size(), 1);
  EXPECT_EQUAL(disc_same_and.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(same_crossing_and,
                                        disc_same_and.point(0), 69.19),
               1.0);
  constexpr auto disc_same_or =
      ad::get_discontinuity_points_and_amplitudes<^^same_crossing_or, 0>(0.37);
  EXPECT_EQUAL(disc_same_or.size(), 1);
  EXPECT_EQUAL(disc_same_or.amplitude(0), 1.0);
  {
    int not_one_jump = 0;
    for (int i = 1; i <= 2000; ++i) {
      const double strike = 0.37 * i;
      const auto both =
          ad::get_discontinuity_points_and_amplitudes_rt<^^same_crossing_and,
                                                         0>(strike);
      const auto either =
          ad::get_discontinuity_points_and_amplitudes_rt<^^same_crossing_or, 0>(
              strike);
      not_one_jump += both.size() != 1 || both.amplitude(0) != 1.0 ||
                      std::fabs(both.point(0) - strike / 1.1) > 1e-13 * strike;
      not_one_jump +=
          either.size() != 1 || either.amplitude(0) != 1.0 ||
          std::fabs(either.point(0) - strike / 1.1) > 1e-13 * strike;
      const auto sum =
          ad::get_discontinuity_points_and_amplitudes_rt<^^same_crossing_sum,
                                                         0>(strike);
      not_one_jump += sum.size() != 1 || sum.amplitude(0) != 2.0 ||
                      std::fabs(sum.point(0) - strike / 1.1) > 1e-13 * strike;
    }
    EXPECT_EQUAL(not_one_jump, 0);
  }
  constexpr auto disc_same_sum =
      ad::get_discontinuity_points_and_amplitudes<^^same_crossing_sum, 0>(
          69.19);
  EXPECT_EQUAL(disc_same_sum.size(), 1);
  EXPECT_EQUAL(disc_same_sum.amplitude(0), 2.0);

  // Every argument but the target, in order
  constexpr auto disc_kfirst =
      ad::get_discontinuity_points_and_amplitudes<^^strike_first_digital, 1>(
          100.0);
  EXPECT_EQUAL(disc_kfirst.size(), 1);
  EXPECT_EQUAL(disc_kfirst.point(0), 100.0);
  EXPECT_EQUAL(disc_kfirst.amplitude(0), 1.0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points<^^strike_first_digital, 1>(100.0)[0]),
      100.0);
  const auto disc_three_rt =
      ad::get_discontinuity_points_and_amplitudes_rt<^^three_arg_digital, 1>(
          90.0, 110.0);
  EXPECT_EQUAL(disc_three_rt.size(), 2);
  EXPECT_EQUAL(disc_three_rt.point(0), 90.0);
  EXPECT_EQUAL(disc_three_rt.amplitude(0), 1.0);
  EXPECT_EQUAL(disc_three_rt.point(1), 110.0);
  EXPECT_EQUAL(disc_three_rt.amplitude(1), -1.0);

  // A value used as a condition is a point, not a jump...
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^spot_as_condition, 0>()
           .size()),
      0);
  EXPECT_EQUAL(spot_as_condition(1e-8) - spot_as_condition(-1e-8), 0.0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^not_spot_minus_strike, 0>(
           100.0)
           .size()),
      0);
  EXPECT_EQUAL(measure_jump_on_function(not_spot_minus_strike, 100.0, 100.0),
               0.0);
  EXPECT_EQUAL((ad::get_discontinuity_points_and_amplitudes<
                    ^^spot_minus_strike_and_strike, 0>(100.0)
                    .size()),
               0);
  EXPECT_EQUAL(
      measure_jump_on_function(spot_minus_strike_and_strike, 100.0, 100.0),
      0.0);
  // ... held at its off-point truth where it shares a point: 0 -> 1, not 5
  constexpr auto disc_truthy =
      ad::get_discontinuity_points_and_amplitudes<^^step_times_truthy_gap, 0>(
          100.0);
  EXPECT_EQUAL(disc_truthy.size(), 1);
  EXPECT_EQUAL(disc_truthy.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(step_times_truthy_gap, 100.0, 100.0),
               disc_truthy.amplitude(0));
  // Stepwise conditions and comparisons follow the comparison inside them
  constexpr auto disc_ind =
      ad::get_discontinuity_points_and_amplitudes<^^indicator_as_condition, 0>(
          100.0);
  EXPECT_EQUAL(disc_ind.size(), 1);
  EXPECT_EQUAL(disc_ind.point(0), 100.0);
  EXPECT_EQUAL(disc_ind.amplitude(0), 2.0);
  EXPECT_EQUAL(measure_jump_on_function(indicator_as_condition, 100.0, 100.0),
               disc_ind.amplitude(0));
  constexpr auto disc_indcmp =
      ad::get_discontinuity_points_and_amplitudes<^^indicator_compared, 0>(
          100.0);
  EXPECT_EQUAL(disc_indcmp.size(), 1);
  EXPECT_EQUAL(disc_indcmp.amplitude(0), 2.0);

  // A pole is an error, not an amplitude (a compile error from the consteval
  // entry points)
  bool pole_rejected = false;
  try {
    (void)ad::get_discontinuity_points_and_amplitudes_rt<
        ^^reciprocal_above_zero, 0>(1.0);
  } catch (const char *) {
    pole_rejected = true;
  }
  EXPECT_TRUE(pole_rejected);

  // So is a crossing whose slope is not finite, not a silently missed jump
  const auto rejected = [](auto analyze) {
    try {
      (void)analyze();
    } catch (const char *) {
      return true;
    }
    return false;
  };
  EXPECT_TRUE(rejected([] {
    return ad::get_discontinuity_points_and_amplitudes_rt<^^ratio_digital, 0>(
        0.0);
  }));
  EXPECT_EQUAL(ratio_digital(1e-9, 0.0) - ratio_digital(-1e-9, 0.0), 1.0);
  EXPECT_TRUE(rejected([] {
    return ad::get_discontinuity_points_and_amplitudes_rt<^^product_digital, 0>(
        std::numeric_limits<double>::infinity());
  }));
  // ... while a finite one, or a flat one, is placed as before
  constexpr auto disc_ratio =
      ad::get_discontinuity_points_and_amplitudes<^^ratio_digital, 0>(2.0);
  EXPECT_EQUAL(disc_ratio.size(), 1);
  EXPECT_EQUAL(disc_ratio.point(0), 2.0);
  EXPECT_EQUAL(disc_ratio.amplitude(0), 1.0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^product_digital, 0>(0.0)
           .size()),
      0);
  // So is one whose sides overflow with the target at 0
  EXPECT_TRUE(rejected([] {
    return ad::get_discontinuity_points_and_amplitudes_rt<^^scaled_gap_digital,
                                                          0>(1e300);
  }));
  EXPECT_EQUAL(scaled_gap_digital(1e300 * (1 + 1e-15), 1e300) -
                   scaled_gap_digital(1e300 * (1 - 1e-15), 1e300),
               1.0);
  EXPECT_TRUE(rejected([] {
    return ad::get_discontinuity_points_and_amplitudes_rt<
        ^^doubled_spot_digital, 0>(1e308);
  }));
  // ... short of the overflow, both are placed as before
  constexpr auto disc_scaled_gap =
      ad::get_discontinuity_points_and_amplitudes<^^scaled_gap_digital, 0>(
          100.0);
  EXPECT_EQUAL(disc_scaled_gap.size(), 1);
  EXPECT_EQUAL(disc_scaled_gap.point(0), 100.0);
  EXPECT_EQUAL(disc_scaled_gap.amplitude(0), 1.0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^doubled_spot_digital, 0>(
           100.0)
           .point(0)),
      100.0);
  // ... while a gap not finite for every target has no point, and no error
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^reciprocal_strike_digital,
                                                   0>(0.0)
           .size()),
      0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^exp_strike_digital, 0>(
           1000.0)
           .size()),
      0);

  // So is a jump whose error cannot be bounded
  bool unbounded_rejected = false;
  try {
    (void)ad::get_discontinuity_points_and_amplitudes_rt<^^rounding_rebate, 0>(
        3.0);
  } catch (const char *) {
    unbounded_rejected = true;
  }
  EXPECT_TRUE(unbounded_rejected);
  EXPECT_TRUE(rounding_rebate(3.0 + 1e-9, 3.0) > 1e15);
  // ... while one through an overflow is measured, at compile time too, as
  // it is short of the overflow
  constexpr auto disc_overflowed =
      ad::get_discontinuity_points_and_amplitudes<^^overflowing_rebate, 0>(
          1000.0);
  EXPECT_EQUAL(disc_overflowed.size(), 1);
  EXPECT_EQUAL(disc_overflowed.point(0), 1000.0);
  EXPECT_EQUAL(disc_overflowed.amplitude(0), 1.0);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^overflowing_rebate, 0>(
           1000.0)
           .amplitude(0)),
      1.0);
  constexpr auto disc_rebate =
      ad::get_discontinuity_points_and_amplitudes<^^overflowing_rebate, 0>(
          100.0);
  EXPECT_EQUAL(disc_rebate.size(), 1);
  EXPECT_NEAR_REL(disc_rebate.amplitude(0), 1.0, 1e-15);

  // More points than MaxPoints is an error, not a silent omission
  bool overflow_rejected = false;
  try {
    (void)ad::get_discontinuity_points_and_amplitudes_rt<
        ^^three_discontinuities, 0, 2>();
  } catch (const char *) {
    overflow_rejected = true;
  }
  EXPECT_TRUE(overflow_rejected);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes<^^three_discontinuities, 0,
                                                   3>()
           .size()),
      3);

  // sin at compile time, and both entry points agree to the bit
  constexpr auto disc_sin =
      ad::get_discontinuity_points_and_amplitudes<^^digital_sin_rebate, 0>(
          100.0);
  EXPECT_EQUAL(disc_sin.size(), 1);
  EXPECT_NEAR_REL(disc_sin.amplitude(0), std::sin(100.0) + 1.0, 1e-15);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^digital_sin_rebate, 0>(
           100.0)
           .amplitude(0)),
      disc_sin.amplitude(0));
  // ... at any strike: the three-part reduction gave sin(1e18) = -2e29
  constexpr auto disc_sin_far =
      ad::get_discontinuity_points_and_amplitudes<^^digital_sin_rebate, 0>(
          1e18);
  EXPECT_EQUAL(disc_sin_far.size(), 1);
  EXPECT_EQUAL(disc_sin_far.point(0), 1e18);
  EXPECT_NEAR_REL(disc_sin_far.amplitude(0), std::sin(1e18) + 1.0, 1e-13);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^digital_sin_rebate, 0>(
           1e18)
           .amplitude(0)),
      disc_sin_far.amplitude(0));
  constexpr auto disc_exp =
      ad::get_discontinuity_points_and_amplitudes<^^digital_exp_payoff, 0>(
          500.0);
  EXPECT_EQUAL(disc_exp.size(), 1);
  EXPECT_NEAR_REL(disc_exp.amplitude(0), std::exp(50.0), 1e-15);
  EXPECT_EQUAL(
      (ad::get_discontinuity_points_and_amplitudes_rt<^^digital_exp_payoff, 0>(
           500.0)
           .amplitude(0)),
      disc_exp.amplitude(0));

  TEST_END;
}
