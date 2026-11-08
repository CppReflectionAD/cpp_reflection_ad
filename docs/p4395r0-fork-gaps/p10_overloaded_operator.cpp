// §3.5: an overloaded operator call is both is_function_call and
// is_operator_expression; expression_operator_of names the operator;
// operands_of gives the two arguments.
#include "probe.h"
struct S {
  double v;
};
S operator+(S, S);
S f(S a, S b) { return a + b; }
constexpr m::info e = m::return_operand_of(stmt<0>(^^f));
static_assert(m::is_function_call(e), "P10 is_function_call");
static_assert(m::is_operator_expression(e), "P10 is_operator_expression");
static_assert(m::expression_operator_of(e) == m::operators::op_plus,
              "P10 op_plus");
static_assert(m::operands_of(e).size() == 2, "P10 operands_of size 2");
