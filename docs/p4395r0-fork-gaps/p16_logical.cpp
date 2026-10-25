// §2.2: logical operators are operator expressions (MY_ASSERT example).
#include "probe.h"
bool in_range(double x, double lo, double hi) {
  return lo <= hi && x >= lo && !(x > hi);
}
constexpr m::info e = m::return_operand_of(stmt<0>(^^in_range));
static_assert(m::is_operator_expression(e), "P16 && is_operator_expression");
static_assert(m::expression_operator_of(e) ==
                  m::operators::op_ampersand_ampersand,
              "P16 op_&&");
constexpr m::info notx = m::operands_of(e)[1];
static_assert(m::is_operator_expression(notx), "P16 ! is_operator_expression");
static_assert(m::expression_operator_of(notx) == m::operators::op_exclamation,
              "P16 op_!");
