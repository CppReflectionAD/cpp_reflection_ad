// is_if_statement and condition_of on an `if`, on both compilers.
//
// The clang fork has is_if_statement, and its condition_of accepts an if
// statement. The GCC fork has no is_if_statement, so ad::meta supplies one:
// GCC reflects an `if` as the same node as `?:`, which its condition_of
// already accepts. A statement that only contains a `?:` is not an if
// statement on either compiler.
#include <ad/meta.h>

namespace m = ad::meta;

consteval m::info strip(m::info e) {
  while (m::is_conversion(e))
    e = m::operands_of(e).front();
  return e;
}

consteval m::info statement(m::info fn, int k) {
  return m::statements_of(m::body_of(fn))[k];
}

double abs_if(double x) {
  if (x > 0.0)
    return x;
  return -x;
}

double abs_if_else(double x) {
  if (x < 0.0)
    return -x;
  else
    return x;
}

double relu_return(double x) { return x > 0.0 ? x : 0.0; }

double relu_assign(double x) {
  double y = 0.0;
  y = x > 0.0 ? x : 0.0;
  return y;
}

double relu_declare(double x) {
  const double y = x > 0.0 ? x : 0.0;
  return y;
}

int calls = 0;
void up() { ++calls; }
void down() { --calls; }

void ternary_statement(double x) { x > 0.0 ? up() : down(); }

static_assert(m::is_if_statement(statement(^^abs_if, 0)));
static_assert(!m::is_if_statement(statement(^^abs_if, 1)));
static_assert(m::is_if_statement(statement(^^abs_if_else, 0)));

// condition_of gives the `x > 0.0` / `x < 0.0` comparison.
constexpr m::info abs_cond = strip(m::condition_of(statement(^^abs_if, 0)));
static_assert(m::is_operator_expression(abs_cond));
static_assert(m::expression_operator_of(abs_cond) == m::operators::op_greater);
static_assert(m::identifier_of(m::declaration_of(
                  strip(m::operands_of(abs_cond)[0]))) == "x");

constexpr m::info abs_else_cond =
    strip(m::condition_of(statement(^^abs_if_else, 0)));
static_assert(m::expression_operator_of(abs_else_cond) ==
              m::operators::op_less);

// A `?:` in a return, an assignment, an initializer or on its own is not an
// if statement.
static_assert(!m::is_if_statement(statement(^^relu_return, 0)));
static_assert(!m::is_if_statement(statement(^^relu_assign, 1)));
static_assert(!m::is_if_statement(statement(^^relu_declare, 0)));
static_assert(!m::is_if_statement(statement(^^ternary_statement, 0)));

int main() { return 0; }
