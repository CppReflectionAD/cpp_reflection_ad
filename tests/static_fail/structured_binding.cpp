#include <ad/forward_derivative.h>

// A structured binding is a declaration statement with no declarators, so
// the engine binds none of its names (#235).
struct Pair {
  double a, b;
};

inline double binding(double x) {
  auto [a, b] = Pair{x, x};
  return a * b;
}

int main() {
  // this should fail to compile with "reflection AD: structured bindings are
  // unsupported"
  [[maybe_unused]] double d = ad::forward_derivative<^^binding, 0>(1.0);
  return 0;
}
