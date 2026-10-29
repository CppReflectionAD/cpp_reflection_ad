#include <ad/forward_derivative.h>

// An assignment can't be lowered: the engine takes straight-line bodies of
// `T v = expr;` declarations and one return. GCC used to report an assignment
// as a declaration statement (#185, fixed by CppReflectionAD/gcc#6), which gave
// this a derivative of 2, not 7, with no error (#189).
inline double assigned(double x) {
  double a = x;
  a = a * a;
  return a + x;
}

int main() {
  // this should fail to compile with "reflection AD: unsupported statement"
  [[maybe_unused]] double d = ad::forward_derivative<^^assigned, 0>(3.0);
  return 0;
}
