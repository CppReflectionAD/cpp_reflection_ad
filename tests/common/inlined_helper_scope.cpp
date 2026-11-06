// Names in an inlined helper resolve to the helper's own parameters and
// locals, never to the caller's. Any other name must be a constexpr variable,
// folded to a constant.
//
// The engine looks names up by identifier, in one environment shared with the
// caller. Before the lookup was bounded to the helper's bindings, on clang
// scale()'s `k` found shadowed()'s local `k` = x^2 instead of the global 2:
// shadowed() computed x^3 + x^2 and gave a derivative of 33 at x = 3, not 8,
// with no error. GCC reflects a read of a constexpr variable as its value, a
// literal, so it never looked `k` up.
#include <test_simple_include.hpp>

#include <ad/forward_derivative.h>
#include <ad/reverse_derivative.h>

constexpr double k = 2.0;

inline double scale(double x) { return k * x; }

inline double shadowed(double x) {
  double k = x * x;
  return scale(x) + k;
}

int main() {
  // 2x + x^2: d/dx = 2 + 2x
  EXPECT_NEAR_ABS((ad::forward_derivative<^^shadowed, 0>(3.0)), 8.0, 1e-12);
  EXPECT_NEAR_ABS((ad::gradient_reverse<^^shadowed>(3.0))[0], 8.0, 1e-12);

  TEST_END;
}
