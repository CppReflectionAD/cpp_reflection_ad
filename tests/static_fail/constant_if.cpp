#include <ad/forward_derivative.h>

// An `if` with a constant condition is a statement of the body (#268), and the
// engine rejects an `if`, so the function is rejected on both compilers. GCC
// used to replace the `if` by its `then` statement, so this function
// differentiated there and not on clang.
inline double square_if(double x) {
  if (true) {
    x = x * x;
  }
  return x;
}

int main() {
  // this should fail to compile: the engine doesn't support an if statement
  [[maybe_unused]] double d = ad::forward_derivative<^^square_if, 0>(3.0);
  return 0;
}
