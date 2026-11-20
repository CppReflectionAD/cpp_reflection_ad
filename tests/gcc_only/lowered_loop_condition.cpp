// Option 2 loop reflection: while, do/while, for, and break remain first-class
// statements rather than lowered labels/gotos/ifs.
#include <ad/meta.h>

namespace m = ad::meta;

consteval m::info strip(m::info e) {
  while (m::is_conversion(e))
    e = m::operands_of(e).front();
  return e;
}

consteval m::info stmt(m::info fn, int k) {
  return m::statements_of(m::body_of(fn))[k];
}

// The operator of a loop's condition.
consteval m::operators cond_op(m::info fn, int k) {
  return m::expression_operator_of(strip(m::condition_of(stmt(fn, k))));
}

double w(double x) {
  double y = x;
  while (y < 10.0)
    y = y * 2.0;
  return y;
}

double dw(double x) {
  double y = x;
  do
    y = y * 2.0;
  while (y < 10.0);
  return y;
}

double f(double x) {
  double y = x;
  for (int i = 0; i < 3; ++i)
    y = y * 2.0;
  return y;
}

double w_break(double x) {
  double y = x;
  while (y < 10.0) {
    if (y > 5.0)
      break;
    y = y * 2.0;
  }
  return y;
}

static_assert(m::is_while_statement(stmt(^^w, 1)));
static_assert(!m::is_if_statement(stmt(^^w, 1)));
static_assert(cond_op(^^w, 1) == m::operators::op_less);

static_assert(m::is_do_statement(stmt(^^dw, 1)));
static_assert(cond_op(^^dw, 1) == m::operators::op_less);

static_assert(m::is_for_statement(stmt(^^f, 1)));
static_assert(m::is_declaration_statement(m::init_statement_of(stmt(^^f, 1))));
static_assert(cond_op(^^f, 1) == m::operators::op_less);
static_assert(m::expression_operator_of(m::increment_of(stmt(^^f, 1))) ==
              m::operators::op_plus_plus);

static_assert(m::is_while_statement(stmt(^^w_break, 1)));
static_assert(cond_op(^^w_break, 1) == m::operators::op_less);
static_assert(m::is_compound_statement(m::loop_body_of(stmt(^^w_break, 1))));
static_assert(m::is_break_statement(m::then_statement_of(
    m::statements_of(m::loop_body_of(stmt(^^w_break, 1)))[0])));

int main() { return 0; }
