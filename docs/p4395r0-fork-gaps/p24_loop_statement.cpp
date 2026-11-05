// §3.2: is_compound_statement identifies a block, and any statement outside the
// proposal's scope can be identified but not decomposed. So a loop isn't a
// compound statement. If a fork does decompose it, every element statements_of
// returns must still satisfy is_statement.
#include "probe.h"
int sink(int);
int while_loop(int n) {
  while (n)
    --n;
  return 0;
}
int for_loop(int n) {
  for (int i = 0; i < n; ++i)
    sink(i);
  return 0;
}
consteval bool all_statements(m::info s) {
  if (!m::is_compound_statement(s))
    return true;
  for (m::info c : m::statements_of(s))
    if (!m::is_statement(c))
      return false;
  return true;
}
static_assert(!m::is_compound_statement(stmt<0>(^^while_loop)),
              "P24 a while is not a compound statement");
static_assert(all_statements(stmt<0>(^^while_loop)),
              "P24 every element of the while's statements_of is a statement");
static_assert(!m::is_compound_statement(stmt<0>(^^for_loop)),
              "P24 a for is not a compound statement");
