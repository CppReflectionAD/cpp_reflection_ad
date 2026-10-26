#include <ad/forward_derivative.h>

// Only an assignment that is a statement of its own rebinds its target; one
// inside an expression is rejected rather than lowered as if it stored
// nothing. (In an initializer, GCC's reflection drops the assignment: see
// known_gap/gcc/assignment_in_initializer.cpp.)
inline double nested(double x) {
  double a = x;
  return (a = a * x) + x;
}

int main() {
  // this should fail to compile with "reflection AD: an assignment inside an
  // expression is not supported"
  [[maybe_unused]] double d = ad::forward_derivative<^^nested, 0>(3.0);
  return 0;
}
