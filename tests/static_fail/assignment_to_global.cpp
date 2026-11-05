#include <ad/forward_derivative.h>

// An assignment rebinds a name of the body being lowered; a global has no
// binding to replace, and its store would have to outlive the call. The
// caller's local `total` doesn't count either: an inlined helper's names
// resolve only to its own parameters and locals.
inline double total = 0.0;

inline double accumulate(double x) {
  total += x;
  return x;
}

inline double tallied(double x) {
  double total = x * x;
  double r = accumulate(x);
  return total + r;
}

int main() {
  // this should fail to compile with "reflection AD: name is not a parameter
  // or local of the reflected body"
  [[maybe_unused]] double d = ad::forward_derivative<^^tallied, 0>(3.0);
  return 0;
}
