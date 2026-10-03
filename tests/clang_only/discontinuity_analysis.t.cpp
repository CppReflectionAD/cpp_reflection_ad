#include "discontinuity_analysis.hpp"
#include <test_simple_include.hpp>

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

// Digital call: spot > K ? 1 : 0
// This has a discontinuity at spot = K with a positive jump (0 -> 1 as spot
// increases) The jump should be +1
double digital_call(double spot, double strike) {
  return (spot > strike) ? 1.0 : 0.0;
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
double strike_lt_spot(double spot, double strike) {
  return (strike < spot) ? 1.0 : 0.0;
}
double strike_plus_one_gt_spot(double spot, double strike) {
  return (strike + 1 > spot) ? 1.0 : 0.0;
}

// Le/Ge
double digital_put_le(double spot, double strike) {
  return (spot <= strike) ? 1.0 : 0.0;
}
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

// Jump of payoff(spot, strike) as spot crosses `point` upward.
double measure_jump_on_function(double (*payoff)(double, double), double point,
                                double strike, double epsilon = 1e-8) {
  return payoff(point + epsilon, strike) - payoff(point - epsilon, strike);
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

  // Test digital_call with amplitudes (verify opposite sign from put)
  // (spot > strike) ? 1.0 : 0.0
  // At strike = 100.0, the jump as spot increases is 1.0 - 0.0 = +1.0.
  constexpr auto disc_call =
      ad::get_discontinuity_points_and_amplitudes<^^digital_call, 0>(100.0);
  EXPECT_FALSE(disc_call.empty());
  EXPECT_EQUAL(disc_call.size(), 1);
  EXPECT_EQUAL(disc_call.point(0), 100.0);
  EXPECT_EQUAL(disc_call.amplitude(0), 1.0);

  // Test that digital put and call have opposite signs (proof of bug fix)
  // If the bug existed, both would report +1 instead of having opposite signs
  // Digital put should be negative, digital call should be positive
  EXPECT_EQUAL(disc_put.amplitude(0) < disc_call.amplitude(0), true);

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

  // Other input on the left, target on the right
  constexpr auto disc_c4 =
      ad::get_discontinuity_points_and_amplitudes<^^strike_lt_spot, 0>(100.0);
  EXPECT_EQUAL(disc_c4.size(), 1);
  EXPECT_EQUAL(disc_c4.point(0), 100.0);
  EXPECT_EQUAL(disc_c4.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(strike_lt_spot, 100.0, 100.0),
               disc_c4.amplitude(0));

  // Binary op on the left, target on the right
  constexpr auto disc_c6 =
      ad::get_discontinuity_points_and_amplitudes<^^strike_plus_one_gt_spot, 0>(
          100.0);
  EXPECT_EQUAL(disc_c6.size(), 1);
  EXPECT_EQUAL(disc_c6.point(0), 101.0);
  EXPECT_EQUAL(disc_c6.amplitude(0), -1.0);
  EXPECT_EQUAL(measure_jump_on_function(strike_plus_one_gt_spot, 101.0, 100.0),
               disc_c6.amplitude(0));

  // Le / Ge
  constexpr auto disc_le =
      ad::get_discontinuity_points_and_amplitudes<^^digital_put_le, 0>(100.0);
  EXPECT_EQUAL(disc_le.amplitude(0), -1.0);
  EXPECT_EQUAL(measure_jump_on_function(digital_put_le, 100.0, 100.0),
               disc_le.amplitude(0));
  constexpr auto disc_ge =
      ad::get_discontinuity_points_and_amplitudes<^^digital_call_ge, 0>(100.0);
  EXPECT_EQUAL(disc_ge.amplitude(0), 1.0);
  EXPECT_EQUAL(measure_jump_on_function(digital_call_ge, 100.0, 100.0),
               disc_ge.amplitude(0));

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

  TEST_END;
}
