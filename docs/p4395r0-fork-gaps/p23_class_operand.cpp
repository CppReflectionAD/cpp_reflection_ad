// §3.5: "Operands can also be other expressions", which are decomposed in turn.
// A class-type argument passed by value still leads to the name it came from.
#include "probe.h"
struct S {
  double v;
};
S operator+(S, S);
S f(S a, S b) { return a + b; }
constexpr m::info call = m::return_operand_of(stmt<0>(^^f));
static_assert(m::operands_of(call).size() == 2, "P23 two operands");
consteval bool reaches_id(m::info e, int depth = 0) {
  if (m::is_id_expression(e))
    return true;
  if (depth == 8)
    return false;
  for (m::info o : m::operands_of(e))
    if (reaches_id(o, depth + 1))
      return true;
  return false;
}
static_assert(reaches_id(m::operands_of(call)[0]),
              "P23 the first operand leads to `a`");
