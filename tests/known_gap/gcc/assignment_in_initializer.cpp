// TEST-PASSES-ON: clang
// Known GCC gap #217: in `double b = (a = a * x);`, GCC reflects b's
// initializer as `a * x`, without the assignment to `a`, and lists no other
// statement for it, so reflection AD can't reject it: on GCC, nested() gets a
// derivative of 2x + 1 (7 at 3), not 4x (12), with no error. Clang reflects
// the assignment, and the engine rejects it as it rejects one in a return on
// both compilers (static_fail/assignment_in_expression.cpp).
//
// When GCC fixes the gap, this file compiles: move it to common/, drop the
// TEST-PASSES-ON line, and delete the .fail.txt. Then nested() fails to
// compile on both compilers, as it should.
#include <ad/meta.h>

namespace m = ad::meta;

inline double nested(double x) {
  double a = x;
  double b = (a = a * x);
  return a + b;
}

consteval m::info initializer_of_b() {
  m::info e = m::initializers_of(m::statements_of(m::body_of(^^nested))[1])[0];
  while (m::is_expression(e) && m::is_conversion(e))
    e = m::operands_of(e).front();
  return e;
}

static_assert(m::is_operator_expression(initializer_of_b()) &&
                  m::expression_operator_of(initializer_of_b()) ==
                      m::operators::op_equals,
              "b's initializer is not the assignment `a = a * x`");

int main() {}
