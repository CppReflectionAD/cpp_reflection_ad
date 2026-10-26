// Assignments `v = e;` and `v op= e;` to a local or a by-value parameter in a
// straight-line body (#213). Each rebinds `v` to the value it stores, so later
// uses of `v` see it and earlier copies keep the old value.
//
// Clang reflects each as an expression statement whose expression is the
// assignment, `+=` and the like included. GCC reflects one as the assignment
// converted to void, with `a += b` already rewritten as `a = a + b`;
// ad::meta::is_expression_statement and expression_of give the clang shape.
// Before #213 the engine rejected every assignment; before #185/#189, GCC took
// `a = a * a;` for a declaration of `a` and gave assigned() a derivative of 2,
// not 7, with no error.
#include <test_simple_include.hpp>

#include <ad/autograd.h>
#include <ad/forward_derivative.h>
#include <ad/meta.h>
#include <ad/reverse_derivative.h>

namespace m = ad::meta;

inline double assigned(double x) {
  double a = x;
  a = a * a;
  return a + x;
}

inline double compound(double x, double y) {
  double s = x * y;
  s += x;
  s *= y;
  return s;
}

inline double sub_div(double x, double y) {
  double d = x;
  d -= y;
  d /= y;
  return d;
}

// A by-value parameter is a local too.
inline double param(double x) {
  x = x * x;
  x += 1.0;
  return x * x;
}

// `b` keeps the value `a` had when `b` was declared.
inline double copy_then_assign(double x) {
  double a = x;
  double b = a;
  a = 2.0 * x;
  return a * b;
}

// Storing a constant cuts the dependence on x.
inline double reset(double x) {
  double a = x * x;
  a = 5.0;
  return a + x;
}

// The helper's `v *= v` rebinds its own parameter, not the caller's `v`.
inline double square_in_place(double v) {
  v *= v;
  return v;
}

inline double caller(double x) {
  double v = x + 1.0;
  double w = square_in_place(x);
  return v * w;
}

consteval m::info statement(m::info fn, std::size_t i) {
  return m::statements_of(m::body_of(fn))[i];
}

static_assert(!m::is_expression_statement(statement(^^assigned, 0)) &&
                  m::is_expression_statement(statement(^^assigned, 1)) &&
                  !m::is_expression_statement(statement(^^assigned, 2)),
              "only the assignment is an expression statement");
static_assert(
    m::is_operator_expression(m::expression_of(statement(^^assigned, 1))) &&
        m::expression_operator_of(m::expression_of(statement(^^assigned, 1))) ==
            m::operators::op_equals,
    "the expression of `a = a * a;` is not the assignment");
static_assert(
    m::is_id_expression(
        m::operands_of(m::expression_of(statement(^^assigned, 1))).front()),
    "the target of `a = a * a;` is not `a`");

int main() {
  // d/dx (x^2 + x) = 2x + 1
  EXPECT_NEAR_ABS((ad::forward_derivative<^^assigned, 0>(3.0)), 7.0, 1e-12);
  EXPECT_NEAR_ABS((ad::gradient_reverse<^^assigned>(3.0))[0], 7.0, 1e-12);

  // (xy + x) y: d/dx = y^2 + y, d/dy = 2xy + x
  EXPECT_NEAR_ABS((ad::forward_derivative<^^compound, 0>(2.0, 3.0)), 12.0,
                  1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^compound, 1>(2.0, 3.0)), 14.0,
                  1e-12);
  auto gc = ad::gradient_reverse<^^compound>(2.0, 3.0);
  EXPECT_NEAR_ABS(gc[0], 12.0, 1e-12);
  EXPECT_NEAR_ABS(gc[1], 14.0, 1e-12);

  // (x - y) / y: d/dx = 1/y, d/dy = -x/y^2
  EXPECT_NEAR_ABS((ad::forward_derivative<^^sub_div, 0>(2.0, 4.0)), 0.25,
                  1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^sub_div, 1>(2.0, 4.0)), -0.125,
                  1e-12);
  auto gs = ad::gradient_reverse<^^sub_div>(2.0, 4.0);
  EXPECT_NEAR_ABS(gs[0], 0.25, 1e-12);
  EXPECT_NEAR_ABS(gs[1], -0.125, 1e-12);

  // (x^2 + 1)^2: d/dx = 4x (x^2 + 1)
  EXPECT_NEAR_ABS((ad::forward_derivative<^^param, 0>(1.5)), 19.5, 1e-12);
  EXPECT_NEAR_ABS((ad::gradient_reverse<^^param>(1.5))[0], 19.5, 1e-12);

  // 2x * x: d/dx = 4x
  EXPECT_NEAR_ABS((ad::forward_derivative<^^copy_then_assign, 0>(3.0)), 12.0,
                  1e-12);
  EXPECT_NEAR_ABS((ad::gradient_reverse<^^copy_then_assign>(3.0))[0], 12.0,
                  1e-12);

  // 5 + x: d/dx = 1
  EXPECT_NEAR_ABS((ad::forward_derivative<^^reset, 0>(3.0)), 1.0, 1e-12);
  EXPECT_NEAR_ABS((ad::gradient_reverse<^^reset>(3.0))[0], 1.0, 1e-12);

  // (x + 1) x^2: d/dx = x^2 + 2x (x + 1)
  EXPECT_NEAR_ABS((ad::forward_derivative<^^caller, 0>(2.0)), 16.0, 1e-12);
  EXPECT_NEAR_ABS((ad::gradient_reverse<^^caller>(2.0))[0], 16.0, 1e-12);

  TEST_END;
}
