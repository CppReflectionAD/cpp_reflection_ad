#include <ad/forward_derivative.h>

// An increment can't be lowered: the engine takes straight-line bodies of
// `T v = expr;` declarations, `v = expr;` and `v op= expr;` assignments, and
// one return. Both compilers reflect `a++;` as the `++` operator, so both
// reject it.
inline double incremented(double x) {
  double a = x;
  a++;
  return a * x;
}

int main() {
  // this should fail to compile with "reflection AD: unsupported statement"
  [[maybe_unused]] double d = ad::forward_derivative<^^incremented, 0>(3.0);
  return 0;
}
