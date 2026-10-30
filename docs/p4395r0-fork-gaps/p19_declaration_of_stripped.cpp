// §3.5: declaration_of(id) == declarators_of(decl)[0], after stripping the
// implicit conversion clang puts around a value use.
#include "probe.h"
double f(double x) {
  double v = x * 2;
  return v + 1;
}
consteval m::info strip(m::info e) {
#ifdef __clang__
  while (m::is_conversion(e))
    e = m::operands_of(e)[0];
#endif
  return e;
}
constexpr m::info ret = m::return_operand_of(stmt<1>(^^f));
constexpr m::info lhs = strip(m::operands_of(ret)[0]);
static_assert(m::is_id_expression(lhs), "P19 stripped `v` is_id_expression");
static_assert(m::declaration_of(lhs) == m::declarators_of(stmt<0>(^^f))[0],
              "P19 declaration_of(stripped v) == declarators_of(decl)[0]");
