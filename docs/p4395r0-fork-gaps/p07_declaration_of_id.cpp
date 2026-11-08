// §3.5: an id-expression operand satisfies is_id_expression, and
// declaration_of gives the variable, matching declarators_of.
#include "probe.h"
double f(double x) {
  double v = x * 2;
  return v + 1;
}
constexpr m::info ret = m::return_operand_of(stmt<1>(^^f));
constexpr m::info lhs = m::operands_of(ret)[0];
static_assert(m::is_id_expression(lhs), "P07 `v` is_id_expression");
static_assert(m::is_expression(lhs), "P07 `v` is_expression");
static_assert(m::declaration_of(lhs) == m::declarators_of(stmt<0>(^^f))[0],
              "P07 declaration_of(v) == declarators_of(decl)[0]");
