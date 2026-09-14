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

  TEST_END;
}
