#include <ad/forward_derivative.h>

// An assignment in a variable's initializer is inside an expression, so it is
// rejected like one in a return (assignment_in_expression.cpp). GCC used to
// leave it out of the reflection, which gave a wrong derivative with no error
// (#217).
inline double nested(double x) {
  double a = x;
  double b = (a = a * x);
  return a + b;
}

int main() {
  // this should fail to compile with "reflection AD: an assignment inside an
  // expression is not supported"
  [[maybe_unused]] double d = ad::forward_derivative<^^nested, 0>(3.0);
  return 0;
}
