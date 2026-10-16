#include <test_simple_include.hpp>

#include <tuple>

// The test framework's own comparisons. A comparison that cannot fail hides
// every bug it is meant to catch, so each is checked in both directions.
int main() {
  using detail_test::expect_near_rel;

  // Near, with the values on either side of zero's sign
  EXPECT_TRUE(std::get<0>(expect_near_rel(1.0, 1.0 + 1e-15, 1e-13)));
  EXPECT_TRUE(std::get<0>(expect_near_rel(-1.0, -1.0 - 1e-15, 1e-13)));
  EXPECT_TRUE(std::get<0>(expect_near_rel(0.0, 0.0, 1e-13)));

  // Far apart: the relative difference is a magnitude, whatever the sign of
  // the values' average (#79)
  EXPECT_FALSE(std::get<0>(expect_near_rel(1.0, 2.0, 1e-13)));
  EXPECT_FALSE(std::get<0>(expect_near_rel(-1.0, -2.0, 1e-13)));
  EXPECT_FALSE(std::get<0>(expect_near_rel(-2e29, 0.007, 1e-13)));
  EXPECT_FALSE(std::get<0>(expect_near_rel(-1.0, 0.5, 1e-13)));
  EXPECT_EQUAL(std::get<1>(expect_near_rel(-1.0, -3.0, 1e-13)), 1.0);

  // Values of opposite sign that average to exactly zero
  EXPECT_FALSE(std::get<0>(expect_near_rel(-1.0, 1.0, 1e-13)));

  TEST_END;
}
