#include <ad/forward_derivative.h>

// A declarator without an initializer can't be lowered: nothing in a
// straight-line body could give it a value later.
inline double no_initializer(double x) {
  [[maybe_unused]] double unused;
  return x;
}

int main() {
  // this should fail to compile with "reflection AD: declarator without an
  // initializer"
  [[maybe_unused]] double d = ad::forward_derivative<^^no_initializer, 0>(1.0);
  return 0;
}
