#include <ad/forward_derivative.h>

// An inlined helper's bindings are dropped when it returns, so a store through
// its `double &` parameter would never reach the caller's `a`: rebinding `r`
// would give scaled() = x^4 + x a derivative of 3x^2 + 1, not 4x^3 + 1, with
// no error.
inline double square_into(double &r, double x) {
  r = x * x;
  return r;
}

inline double scaled(double x) {
  double a = x;
  double s = square_into(a, x);
  return a * s + x;
}

int main() {
  // this should fail to compile with "reflection AD: cannot assign through a
  // reference"
  [[maybe_unused]] double d = ad::forward_derivative<^^scaled, 0>(3.0);
  return 0;
}
