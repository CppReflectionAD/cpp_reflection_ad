// §3.6: "We propose extending this enum to add all possible operators, such as
// the conditional operator". Is ?: an operator expression with an enum value?
#include "probe.h"
int max(int x, int y) { return (x > y) ? x : y; }
constexpr m::info e = m::return_operand_of(stmt<0>(^^max));
static_assert(m::is_operator_expression(e), "P14b ?: is_operator_expression");
constexpr auto op = m::expression_operator_of(e);
