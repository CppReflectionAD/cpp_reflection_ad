#include <ad/forward_derivative.h>

// A local declared without an initializer and assigned later is rejected like
// any declarator without one (declarator_without_initializer.cpp): give it its
// value where it is declared. GCC used to take the later assignment for its
// initializer and accept this (#222).
inline double assigned_later(double x) {
  double t;
  t = x * x;
  return t;
}

int main() {
  // this should fail to compile with "reflection AD: declarator without an
  // initializer"
  [[maybe_unused]] double d = ad::forward_derivative<^^assigned_later, 0>(3.0);
  return 0;
}
