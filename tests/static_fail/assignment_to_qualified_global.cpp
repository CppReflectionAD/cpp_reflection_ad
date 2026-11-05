#include <ad/forward_derivative.h>

// `::total` is the global, not the local of the same name, so this stores to
// the global and returns x. An assignment rebinds a name of the body being
// lowered, and a global has no binding to replace. The engine looks names up by
// identifier, so before the lookup was limited to declarations whose parent is
// a function, it rebound the local: on both compilers this function computed
// 2x, with a derivative of 2 instead of 1, and no error.
inline double total = 0.0;

inline double qualified(double x) {
  double total = x;
  ::total += x;
  return total;
}

int main() {
  // this should fail to compile with "reflection AD: name is not a parameter
  // or local of the reflected body"
  [[maybe_unused]] double d = ad::forward_derivative<^^qualified, 0>(3.0);
  return 0;
}
