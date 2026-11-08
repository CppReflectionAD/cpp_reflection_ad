// §3.5: declaration_of(id) == declarators_of(decl)[0], after stripping any
// implicit conversion around a value use (clang has one here, GCC doesn't).
#include "probe.h"
double f(double x) {
  double v = x * 2;
  return v + 1;
}
consteval m::info strip(m::info e) {
  while (m::is_conversion(e))
    e = m::operands_of(e)[0];
  return e;
}
constexpr m::info ret = m::return_operand_of(stmt<1>(^^f));
constexpr m::info lhs = strip(m::operands_of(ret)[0]);
static_assert(m::is_id_expression(lhs), "P19 stripped `v` is_id_expression");
static_assert(m::declaration_of(lhs) == m::declarators_of(stmt<0>(^^f))[0],
              "P19 declaration_of(stripped v) == declarators_of(decl)[0]");
