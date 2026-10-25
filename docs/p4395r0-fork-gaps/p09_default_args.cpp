// §3.5: operands_of on a call returns one operand per parameter, including
// defaulted arguments (and not the callee).
#include "probe.h"
double h(double, double = 2.0);
double f(double x) { return h(x); }
constexpr m::info call = m::return_operand_of(stmt<0>(^^f));
static_assert(m::operands_of(call).size() == 2,
              "P09 operands_of(h(x)) has 2 operands");
static_assert(m::is_id_expression(m::operands_of(call)[0]),
              "P09 first operand is x");
