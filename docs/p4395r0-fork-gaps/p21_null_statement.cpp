// §3.2 (#237): statements_of returns the statements as written, and in
// [stmt.expr] a null statement is an expression statement with no expression.
#include "probe.h"
double f(double x) {
  ;
  return x;
}
static_assert(nstmts(^^f) == 2, "P21 the null statement is a statement");
static_assert(m::is_expression_statement(stmt<0>(^^f)),
              "P21 a null statement is_expression_statement");
