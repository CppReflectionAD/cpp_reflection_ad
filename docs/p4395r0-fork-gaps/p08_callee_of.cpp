// §3.5: callee_of is the raw callee expression; resolve it with
// is_id_expression + declaration_of.
#include "probe.h"
double g(double);
double f(double x) { return g(x); }
constexpr m::info call = m::return_operand_of(stmt<0>(^^f));
static_assert(m::is_function_call(call), "P08 is_function_call");
static_assert(m::is_expression(m::callee_of(call)),
              "P08 callee_of is_expression");
static_assert(m::is_id_expression(m::callee_of(call)),
              "P08 callee_of is_id_expression");
static_assert(m::declaration_of(m::callee_of(call)) == ^^g,
              "P08 declaration_of(callee) == ^^g");
