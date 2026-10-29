// A computed argument bound to a `const T&` parameter, as `x - y` is in
// `std::max(x - y, 0.5)`, is that expression once its conversions are peeled
// off.
//
// Both compilers reflect the temporary as an implicit conversion of the
// expression. GCC used to reflect it as a construct whose operands were the
// temporary's unnamed variable and then the expression, and ad/meta.h dropped
// the variable (#199). CppReflectionAD/gcc#11 made it an untyped temporary
// whose operands_of skipped the `-` (#226). Either way, reflection AD on GCC
// took `x - y` for a constant and failed with "'x' is not a constant
// expression". A named variable or a literal argument was already fine.
#include <test_simple_include.hpp>

#include <ad/autograd.h>
#include <ad/forward_derivative.h>
#include <ad/meta.h>
#include <ad/reverse_derivative.h>

#include <algorithm>
#include <cmath>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

double f_diff(double x, double y) { return std::max(x - y, 0.5); }
double f_neg(double x) { return std::min(-x, 0.5); }
double f_sin(double x) { return std::max(std::sin(x), 0.5); }
double f_nested(double x, double y, double z) {
  return std::max(std::max(x - y, 0.5), z);
}

// The call to std::max, below GCC's `*` for its `const T&` return.
consteval m::info call_of(m::info fn) {
  m::info e = return_expr(fn);
  while (!m::is_function_call(e))
    e = m::operands_of(e).front();
  return e;
}

consteval m::info first_argument(m::info fn) {
  m::info e = m::operands_of(call_of(fn)).front();
  while (m::is_conversion(e))
    e = m::operands_of(e).front();
  return e;
}

static_assert(m::is_operator_expression(first_argument(^^f_diff)) &&
                  m::expression_operator_of(first_argument(^^f_diff)) ==
                      m::operators::op_minus,
              "x - y is not an operator expression below its conversions");

int main() {
  // Each on both sides of the kink.
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_diff, 0>(2.0, 0.5)), 1.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_diff, 1>(2.0, 0.5)), -1.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_diff, 0>(0.8, 0.5)), 0.0, 1e-12);

  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_neg, 0>(1.0)), -1.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_neg, 0>(-1.0)), 0.0, 1e-12);

  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_sin, 0>(1.2)), std::cos(1.2),
                  1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_sin, 0>(0.1)), 0.0, 1e-12);

  auto g = ad::gradient_reverse<^^f_nested>(2.0, 0.5, 1.0);
  EXPECT_NEAR_ABS(g[0], 1.0, 1e-12);
  EXPECT_NEAR_ABS(g[1], -1.0, 1e-12);
  EXPECT_NEAR_ABS(g[2], 0.0, 1e-12);
  auto h = ad::gradient_reverse<^^f_nested>(2.0, 0.5, 3.0);
  EXPECT_NEAR_ABS(h[0], 0.0, 1e-12);
  EXPECT_NEAR_ABS(h[1], 0.0, 1e-12);
  EXPECT_NEAR_ABS(h[2], 1.0, 1e-12);

  TEST_END;
}
