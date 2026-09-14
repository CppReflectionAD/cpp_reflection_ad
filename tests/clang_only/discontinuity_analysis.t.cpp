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
  return ((x > 1.0) ? 1.0 : 0.0) + ((x > 4.0) ? 1.0 : 0.0) +
         ((x > 7.0) ? 1.0 : 0.0);
}

// Multi-argument function: 1 discontinuity w.r.t. first argument, second fixed
// Simple case: target is arg 0, fixed arg 1
double two_arg_compare(double x, double y) { return (x > y) ? 1.0 : 0.0; }

int main() {
  // Test 0 discontinuities
  constexpr auto disc0 = ad::get_discontinuity_points<^^continuous_linear, 0>();
  // Should have no points (all zeros)
  EXPECT_EQUAL(disc0[0], 0.0);

  // Test 1 discontinuity
  constexpr auto disc1 = ad::get_discontinuity_points<^^one_discontinuity, 0>();
  EXPECT_EQUAL(disc1[0], 3.0);
  EXPECT_EQUAL(disc1[1], 0.0); // second point should be zero

  // Test 2 discontinuities
  constexpr auto disc2 =
      ad::get_discontinuity_points<^^two_discontinuities, 0>();
  EXPECT_EQUAL(disc2[0], 2.0);
  EXPECT_EQUAL(disc2[1], 5.0);
  EXPECT_EQUAL(disc2[2], 0.0); // third point should be zero

  // Test 3 discontinuities
  constexpr auto disc3 =
      ad::get_discontinuity_points<^^three_discontinuities, 0>();
  EXPECT_EQUAL(disc3[0], 1.0);
  EXPECT_EQUAL(disc3[1], 4.0);
  EXPECT_EQUAL(disc3[2], 7.0);
  EXPECT_EQUAL(disc3[3], 0.0); // fourth point should be zero

  // Test multi-argument function
  // two_arg_compare(x, y) where x is target (arg 0) and y is fixed at 5.0
  // Discontinuity at x = y = 5.0
  constexpr auto disc_2arg =
      ad::get_discontinuity_points<^^two_arg_compare, 0>(5.0);
  EXPECT_EQUAL(disc_2arg[0], 5.0);
  EXPECT_EQUAL(disc_2arg[1], 0.0);

  TEST_END;
}
