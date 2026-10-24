// `||` and `&&` are binary operators and `!` is a unary operator
// (is_operator_expression == true, with operands_of().size() giving the arity).
//
// clang only, because GCC classifies all three wrongly: see
// tests/known_gap/gcc/logical_operator_reflection.cpp, which holds the same
// asserts. When GCC fixes the gap, move this file to tests/common/ and delete
// that one.
#include <meta>

#include "reflection/return_expr.hpp"

namespace m = std::meta;

bool f_or(double x) { return (x > 0.0) || (x > 1.0); }
bool f_and(double x) { return (x > 0.0) && (x > 1.0); }
bool f_not(double x) { return !(x > 0.0); }

static_assert(m::is_operator_expression(return_expr(^^f_or)) &&
                  m::operands_of(return_expr(^^f_or)).size() == 2,
              "|| is not a binary operator");
static_assert(m::is_operator_expression(return_expr(^^f_and)) &&
                  m::operands_of(return_expr(^^f_and)).size() == 2,
              "&& is not a binary operator");
static_assert(m::is_operator_expression(return_expr(^^f_not)) &&
                  m::operands_of(return_expr(^^f_not)).size() == 1,
              "! is not a unary operator");

int main() {}
