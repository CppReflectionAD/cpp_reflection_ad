// condition_of gives a loop's condition on GCC, through the loop's lowered if.
//
// GCC lowers `while`, `do`/`while` and `for` before reflection sees the body
// (#247): the loop becomes a compound statement of labels, gotos and an
// `if (cond) goto body; else goto exit;`. ad::meta's is_if_statement is true
// for that if, and condition_of on it gives the loop's condition. A `break`
// lowers to an if too, `if (cond) goto exit;`. The clang fork keeps the loop,
// and its condition_of accepts the loop itself. See
// docs/gcc-loop-lowering.md.
#include <ad/meta.h>
#include <vector>

namespace m = ad::meta;

consteval m::info strip(m::info e) {
  while (m::is_conversion(e))
    e = m::operands_of(e).front();
  return e;
}

// The if statements in `s`, a compound statement, and in the compound
// statements nested in it, in order. It doesn't look inside an if.
consteval void collect_ifs(m::info s, std::vector<m::info> &r) {
  for (m::info c : m::statements_of(s))
    if (m::is_if_statement(c))
      r.push_back(c);
    else if (m::is_compound_statement(c))
      collect_ifs(c, r);
}

// The if statements in a function's `k`th statement.
consteval std::vector<m::info> ifs_in(m::info fn, int k) {
  std::vector<m::info> r;
  collect_ifs(m::statements_of(m::body_of(fn))[k], r);
  return r;
}

// The operator of the condition of the `i`th of those ifs.
consteval m::operators cond_op(m::info fn, int k, int i) {
  return m::expression_operator_of(strip(m::condition_of(ifs_in(fn, k)[i])));
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

// The loop is not an if statement; the if inside it is.
static_assert(m::is_compound_statement(m::statements_of(m::body_of(^^w))[1]));
static_assert(!m::is_if_statement(m::statements_of(m::body_of(^^w))[1]));

static_assert(ifs_in(^^w, 1).size() == 1);
static_assert(cond_op(^^w, 1, 0) == m::operators::op_less);

static_assert(ifs_in(^^dw, 1).size() == 1);
static_assert(cond_op(^^dw, 1, 0) == m::operators::op_less);

// The for's init-statement is in a block around the lowered loop, and the
// condition reads the variable it declares.
static_assert(ifs_in(^^f, 1).size() == 1);
static_assert(cond_op(^^f, 1, 0) == m::operators::op_less);
static_assert(m::identifier_of(m::declaration_of(strip(m::operands_of(
                  strip(m::condition_of(ifs_in(^^f, 1)[0])))[0]))) == "i");

// The break's if comes first, then the loop's.
static_assert(ifs_in(^^w_break, 1).size() == 2);
static_assert(cond_op(^^w_break, 1, 0) == m::operators::op_greater);
static_assert(cond_op(^^w_break, 1, 1) == m::operators::op_less);

int main() { return 0; }
