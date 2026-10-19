// Uses the installed engine through reflection_ad::ad alone: if the target
// didn't carry the reflection flags, the fork's standard library and the
// include path, this wouldn't compile, link or run.

#include <ad/forward_derivative.h>
#include <ad/reverse_derivative.h>

#include <cx_std/cx_exp.hpp>

#include <cmath>
#include <cstdio>

inline double f(double x, double y) { return x * x * y + std::exp(y); }

int main() {
  double const x = 2.0;
  double const y = 0.5;
  double const df_dx = 2 * x * y;
  double const df_dy = x * x + std::exp(y);

  auto const fwd = ad::forward_derivative<^^f, 0>(x, y);
  auto const rev = ad::gradient_reverse<^^f>(x, y);
  static_assert(cx::exp(0.0) == 1.0);

  bool const ok = std::abs(fwd - df_dx) < 1e-12 &&
                  std::abs(rev[0] - df_dx) < 1e-12 &&
                  std::abs(rev[1] - df_dy) < 1e-12;
  std::printf("df/dx forward %.17g reverse %.17g, df/dy reverse %.17g: %s\n",
              fwd, rev[0], rev[1], ok ? "ok" : "WRONG");
  return ok ? 0 : 1;
}
