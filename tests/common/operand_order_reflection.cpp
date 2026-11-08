// operands_of gives an operator's operands in source order, and an expression
// is reflected as written, on both compilers: `d + p` is `+` of d then p, and
// `!(a || b)` is `!` of `a || b`.
//
// GCC used to reflect the body after folding it, so the operands of `d + p`,
// `d * h(e)` and `2.0 * d` came back swapped, `x - x` was the literal 0 and
// `!(a || b)` was `!a && !b` (#224). Fixed in CppReflectionAD/gcc#22. The fork
// has the same checks in g++.dg/reflect/operand_order*.C.
#include <ad/meta.h>

#include <string_view>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

double h(double y) { return y; }

double r_plus(double d, int p) { return d + p; }
double r_call(double d, double e) { return d * h(e); }
double r_literal(double d) { return 2.0 * d; }
int r_self(int x) { return x - x; }
bool r_not(bool a, bool b) { return !(a || b); }
double i_plus(double d, int p) {
  double q = d + p;
  return q;
}

consteval m::info strip(m::info e) {
  while (m::is_conversion(e))
    e = m::operands_of(e)[0];
  return e;
}
consteval m::info operand(m::info e, int i) {
  return strip(m::operands_of(e)[i]);
}
consteval bool names(m::info e, std::string_view name) {
  return m::is_id_expression(e) &&
         m::identifier_of(m::declaration_of(e)) == name;
}

constexpr m::info PLUS = return_expr(^^r_plus);
constexpr m::info CALL = return_expr(^^r_call);
constexpr m::info LITERAL = return_expr(^^r_literal);
constexpr m::info SELF = return_expr(^^r_self);
constexpr m::info NOT = return_expr(^^r_not);
constexpr m::info INIT =
    strip(m::initializers_of(m::statements_of(m::body_of(^^i_plus))[0])[0]);

static_assert(names(operand(PLUS, 0), "d") && names(operand(PLUS, 1), "p"),
              "d + p: the operands are not d then p");
static_assert(names(operand(CALL, 0), "d") &&
                  m::is_function_call(operand(CALL, 1)),
              "d * h(e): the operands are not d then the call");
static_assert(m::is_literal(operand(LITERAL, 0)) &&
                  names(operand(LITERAL, 1), "d"),
              "2.0 * d: the operands are not 2.0 then d");
static_assert(m::is_operator_expression(SELF) &&
                  m::expression_operator_of(SELF) == m::operators::op_minus,
              "x - x: not reflected as a subtraction");
static_assert(m::expression_operator_of(NOT) == m::operators::op_exclamation &&
                  m::expression_operator_of(operand(NOT, 0)) ==
                      m::operators::op_pipe_pipe,
              "!(a || b): not reflected as ! of a || b");
static_assert(names(operand(INIT, 0), "d") && names(operand(INIT, 1), "p"),
              "double q = d + p: the operands are not d then p");

int main() {}
