// Known clang gap: a built-in subscript `a[i]` has no operator.
//
// P4395 §3.5 and §3.6: expression_operator_of gives the operator of an
// operator expression, from std::meta::operators, which has
// op_square_brackets. The clang fork's expression_operator_of only maps
// built-in unary and binary operators (UnaryOperator, BinaryOperator), so a
// built-in subscript (ArraySubscriptExpr) gives no operator, and no public
// predicate recognises it either. ihm_expressions.hpp prints such a
// subscript through gap::expression_operator_of in
// include/ihm/clang_only/gap.hpp, which recognises it by the fork's
// private expression kind number.
//
// The fix in the fork is one line in expression_operator_of
// (clang/lib/AST/ExprConstantMeta.cpp): map ArraySubscriptExpr to
// OO_Subscript.
//
// This file expects the error recorded in builtin_subscript_operator.fail.txt.
// When the fork fixes the gap, move this file to tests/clang_only/, delete the
// .fail.txt, and replace gap::expression_operator_of with
// std::meta::expression_operator_of (deleting it from gap.hpp).
#include <meta>

namespace m = std::meta;

const double values[2] = {1.0, 2.0};
const double builtin = values[1];

// The initializer of `var`, without its conversions.
consteval m::info initializer(m::info var) {
  m::info e = m::initializer_of(var);
  while (m::is_cast(e))
    e = m::operands_of(e)[0];
  return e;
}

// `values[1]` is found (it has two operands, the array and the index) but has
// no operator.
static_assert(m::operands_of(initializer(^^builtin)).size() == 2);
static_assert(m::expression_operator_of(initializer(^^builtin)) ==
              m::op_square_brackets);

int main() {}
