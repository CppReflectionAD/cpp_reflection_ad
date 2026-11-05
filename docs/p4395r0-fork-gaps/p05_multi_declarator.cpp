// §3.3 + §5.1 (`double v = expr, w = expr2;`): one declaration statement with
// two declarators; declarators_of/initializers_of each have two elements and
// the initializers are expressions.
#include "probe.h"
double f(double x) {
  double v = x * 2, w = v + 1;
  return v * w;
}
static_assert(nstmts(^^f) == 2, "P05 two statements (decl, return)");
static_assert(m::is_declaration_statement(stmt<0>(^^f)),
              "P05 is_declaration_statement");
static_assert(m::is_simple_declaration(stmt<0>(^^f)),
              "P05 is_simple_declaration");
static_assert(m::declarators_of(stmt<0>(^^f)).size() == 2,
              "P05 declarators_of size 2");
static_assert(m::initializers_of(stmt<0>(^^f)).size() == 2,
              "P05 initializers_of size 2");
static_assert(m::is_expression(m::initializers_of(stmt<0>(^^f))[1]),
              "P05 initializer is_expression");
