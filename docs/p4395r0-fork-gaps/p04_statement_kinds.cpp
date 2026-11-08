// §3.2: every statements_of element satisfies is_statement, including
// out-of-scope kinds (for); a nested block is_compound_statement and
// statements_of decomposes it; an expression statement is
// is_expression_statement.
#include "probe.h"
int sink(int);
int f(int n) {
  int s = 0;
  {
    sink(1);
    sink(2);
  }
  for (int i = 0; i < n; ++i)
    sink(i);
  sink(3);
  return s;
}
static_assert(nstmts(^^f) == 5, "P04 five statements");
static_assert(m::is_statement(stmt<0>(^^f)) && m::is_statement(stmt<1>(^^f)) &&
                  m::is_statement(stmt<2>(^^f)) &&
                  m::is_statement(stmt<3>(^^f)) &&
                  m::is_statement(stmt<4>(^^f)),
              "P04 all are is_statement");
static_assert(m::is_compound_statement(stmt<1>(^^f)),
              "P04 nested block is_compound_statement");
static_assert(m::statements_of(stmt<1>(^^f)).size() == 2,
              "P04 statements_of(nested block) == 2");
static_assert(m::is_expression_statement(stmt<3>(^^f)),
              "P04 sink(3); is_expression_statement");
static_assert(m::is_return_statement(stmt<4>(^^f)),
              "P04 return is_return_statement");
