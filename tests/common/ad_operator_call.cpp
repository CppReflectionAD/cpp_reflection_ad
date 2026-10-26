// ad_operator_call.cpp — overloaded operators in a reflected scalar body.
//
// An overloaded operator is a call to its operator function, so the engine
// lowers it as one: it inlines the function's body, whatever its symbol
// usually means. That covers function objects and lambdas (`operator()`),
// `operator[]`, and binary operators such as an `operator^` that squares.
//
// A member operator's object is its first operand on both compilers (`f` in
// `f(x)`), but has no parameter; the engine leaves it out when inlining.

#include <test_simple_include.hpp>

#include <ad/autograd.h>
#include <ad/forward_derivative.h>
#include <ad/meta.h>
#include <ad/reverse_derivative.h>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

struct Square {
  constexpr double operator()(double v) const { return v * v; }
};
inline constexpr Square square{};

struct Cube {
  constexpr double operator[](double v) const { return v * v * v; }
};
inline constexpr Cube cube{};

struct Squared {};
inline constexpr Squared squared{};
constexpr double operator^(double v, [[maybe_unused]] Squared s) {
  return v * v;
}

constexpr double f_functor(double x) { return square(x); }
constexpr double f_subscript(double x) { return cube[x]; }
constexpr double f_caret(double x) { return x ^ squared; }
constexpr double f_lambda(double x, double y) {
  auto mul = [](double a, double b) { return a * b; };
  return mul(x, y) + x;
}
constexpr double f_generic_lambda(double x) {
  const auto sq = [](auto v) { return v * v; };
  return sq(x);
}

// The object is the first operand, and the call is a function call on both
// compilers. (Only clang's is_operator_expression is also true for it.)
static_assert(m::is_function_call(return_expr(^^f_functor)));
static_assert(m::operands_of(return_expr(^^f_functor)).size() == 2);
static_assert(m::operands_of(return_expr(^^f_subscript)).size() == 2);
static_assert(m::operands_of(return_expr(^^f_caret)).size() == 2);

int main() {
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_functor, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_subscript, 0>(2.0)), 12.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_caret, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_generic_lambda, 0>(3.0)), 6.0,
                  1e-12);

  auto g = ad::gradient_reverse<^^f_lambda>(3.0, 5.0);
  EXPECT_NEAR_ABS(g[0], 6.0, 1e-12); // y + 1
  EXPECT_NEAR_ABS(g[1], 3.0, 1e-12); // x

  auto gc = ad::gradient_reverse<^^f_caret>(3.0);
  EXPECT_NEAR_ABS(gc[0], 6.0, 1e-12);

  TEST_END;
}
