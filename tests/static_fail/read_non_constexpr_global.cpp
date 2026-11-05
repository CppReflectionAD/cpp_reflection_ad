#include <ad/forward_derivative.h>

// A name that isn't a parameter or local is folded to a constant, which needs
// its value at compile time. A non-constexpr global has none, so `rate * x`
// can't be differentiated.
inline double rate = 0.05;

inline double grown(double x) { return rate * x; }

int main() {
  // this should fail to compile: `rate` is not usable in a constant expression
  [[maybe_unused]] double d = ad::forward_derivative<^^grown, 0>(3.0);
  return 0;
}
