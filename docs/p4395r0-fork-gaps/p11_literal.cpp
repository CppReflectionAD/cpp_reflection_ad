// §3.5: is_literal on a literal operand; built-in binary operator.
#include "probe.h"
double f(double x) { return x * 2.0; }
constexpr m::info e = m::return_operand_of(stmt<0>(^^f));
static_assert(m::is_expression(e), "P11 is_expression");
static_assert(m::is_operator_expression(e), "P11 is_operator_expression");
static_assert(!m::is_function_call(e), "P11 built-in * is not a call");
static_assert(m::expression_operator_of(e) == m::operators::op_star,
              "P11 op_star");
static_assert(m::is_literal(m::operands_of(e)[1]), "P11 2.0 is_literal");
