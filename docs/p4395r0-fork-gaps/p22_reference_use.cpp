// §3.5: an operand that names a reference is an identifier, so it satisfies
// is_id_expression, the same as one naming any other variable.
#include "probe.h"
void f(int &i) { ++i; }
constexpr m::info inc = m::expression_of(stmt<0>(^^f));
static_assert(m::is_id_expression(m::operands_of(inc)[0]),
              "P22 a use of a reference is_id_expression");
