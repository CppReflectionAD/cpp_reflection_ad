// In `double b = (a = a * x);`, b's initializer is the assignment `a = a * x`
// on both compilers, so reflection AD sees it and rejects nested(): see
// static_fail/assignment_in_initializer.cpp. GCC used to reflect the
// initializer as `a * x`, which gave nested() a wrong derivative with no
// error (#217).
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
