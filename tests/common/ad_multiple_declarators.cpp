// ad_multiple_declarators.cpp — a declaration statement with several
// declarators in a reflected body, including one whose initializer uses an
// earlier declarator of the same statement.
//
// clang reflects `T a = x, b = a * a;` as one statement with two declarators;
// GCC reflects it as two statements. Either way the engine lowers each
// declarator in order.

#include <test_simple_include.hpp>

#include <ad/autograd.h>
#include <ad/forward_derivative.h>
#include <ad/reverse_derivative.h>

constexpr double chained(double x) {
  const double a = x, b = a * a;
  return a + b; // x + x^2
}

constexpr double constants(double x, double y) {
  const double a = 2, b = 3, c = x * y;
  return a * x + b * y + c;
}

int main() {
  EXPECT_NEAR_ABS((ad::forward_derivative<^^chained, 0>(3.0)), 7.0, 1e-12);

  auto g = ad::gradient_reverse<^^constants>(3.0, 5.0);
  EXPECT_NEAR_ABS(g[0], 2.0 + 5.0, 1e-12); // a + y
  EXPECT_NEAR_ABS(g[1], 3.0 + 3.0, 1e-12); // b + x

  TEST_END;
}
