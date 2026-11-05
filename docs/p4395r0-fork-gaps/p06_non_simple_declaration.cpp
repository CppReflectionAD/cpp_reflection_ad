// §3.3: a structured binding and an alias declaration are declaration
// statements but not simple declarations.
#include "probe.h"
struct P {
  double a, b;
};
double f(P p) {
  auto [a, b] = p;
  using D = double;
  return D(a) + b;
}
static_assert(nstmts(^^f) == 3, "P06 three statements");
static_assert(m::is_declaration_statement(stmt<0>(^^f)),
              "P06 binding is_declaration_statement");
static_assert(!m::is_simple_declaration(stmt<0>(^^f)),
              "P06 binding !is_simple_declaration");
static_assert(m::is_declaration_statement(stmt<1>(^^f)),
              "P06 alias is_declaration_statement");
static_assert(!m::is_simple_declaration(stmt<1>(^^f)),
              "P06 alias !is_simple_declaration");
