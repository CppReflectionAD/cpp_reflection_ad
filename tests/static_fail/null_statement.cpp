#include <ad/forward_derivative.h>

// A null statement is a statement of the body (#237) with no expression, so the
// engine's expression_of throws on it, and the function is rejected on both
// compilers. GCC used to leave the statement out, so this function
// differentiated there and not on clang.
inline double stray(double x) {
  ;
  return x * x;
}

int main() {
  // this should fail to compile: a null statement has no expression
  [[maybe_unused]] double d = ad::forward_derivative<^^stray, 0>(3.0);
  return 0;
}
