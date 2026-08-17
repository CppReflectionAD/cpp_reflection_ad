// ad_scalar_demo.cpp — scalar AD on reflected functions: forward mode, reverse
// mode, and activity analysis. Pick an ordinary function; its derivative is
// generated at compile time and spliced as inlined arithmetic (no DSL, no tape).
//
// Build/run: `make run-scalar`.

#include "autograd.h"
#include <cmath>
#include <cstdio>

// Ordinary functions, differentiated by reflection without modification.
constexpr double poly(double x)    { return x * x + 2.0 * x; }        // f' = 2x+2
constexpr double trig(double x)    { return std::sin(x * x); }        // f' = 2x cos(x^2)
constexpr double two_arg(double x, double y) { return x * y + std::exp(x); }  // ∇=[y+e^x, x]
// multi-assignment with a shared intermediate (a DAG, not a tree):
constexpr double shared(double x, double y) {
  double a = -x * -y;                // a is used twice. Test - unary operator too.
  double b = +std::sin(a);           // test + unary operator here too
  return a * b;                      // f = (xy) sin(xy)
}

static int failures = 0;
static void check(const char *name, double got, double want) {
  bool ok = std::abs(got - want) < 1e-6 * (1.0 + std::abs(want));
  std::printf("  %-22s %+.6f (want %+.6f) %s\n", name, got, want, ok ? "ok" : "FAIL");
  failures += !ok;
}

int main() {
  std::printf("=== scalar AD on reflected functions ===\n");

  std::printf("forward mode  (one directional derivative):\n");
  check("poly'(3)",   ad::forward_derivative<^^poly, 0>(3.0), 2*3.0 + 2);
  { double x = 1.3;
    check("trig'(1.3)", ad::forward_derivative<^^trig, 0>(x), 2*x*std::cos(x*x)); }

  std::printf("reverse mode  (whole gradient in one pass):\n");
  { double x = 0.7, y = 1.9, p = x*y;
    auto g = ad::gradient_reverse<^^shared>(x, y);   // shared 'a' handled once
    check("d/dx (xy)sin(xy)", g[0], y*std::sin(p) + p*std::cos(p)*y);
    check("d/dy (xy)sin(xy)", g[1], x*std::sin(p) + p*std::cos(p)*x); }
  { double x = 0.5, y = 2.0;                          // reverse == forward gradient
    auto gr = ad::gradient_reverse<^^two_arg>(x, y);
    auto gf = ad::gradient_of<^^two_arg>(x, y);
    check("rev==fwd d/dx", gr[0], gf[0]);
    check("rev==fwd d/dy", gr[1], gf[1]); }

  // Activity analysis: d/dy of (x*y + exp(x)) is just x — the exp term is pruned
  // (no exp call, no x*0); verified as code quality in `make run-bench`.
  check("d/dy prunes exp", ad::forward_derivative<^^two_arg, 1>(0.5, 2.0), 0.5);

  std::printf("higher-order (differentiate the DAG recursively):\n");
  { 
    double x = 1.7;
    check("poly'(x)",  ad::partial_derivative<^^poly, 0>(x), 2*x + 2.0);
    check("poly''(x)",  ad::partial_derivative<^^poly, 0, 0>(x), 2.0);
    check("poly'''(x)", ad::partial_derivative<^^poly, 0, 0, 0>(x), 0.0);
    // sin(x^2)'' = 2cos(x^2) - 4x^2 sin(x^2)
    check("trig''(x)",  ad::partial_derivative<^^trig, 0, 0>(x),
          2*std::cos(x*x) - 4*x*x*std::sin(x*x));
  }
  { 
    double x = 0.5, y = 2.0;                            // two_arg = xy + e^x
    check("d2/dxdy",    ad::partial_derivative<^^two_arg, 0, 1>(x, y), 1.0);
    check("d2/dx2",     ad::partial_derivative<^^two_arg, 0, 0>(x, y), std::exp(x));
    check("d2/dy2",     ad::partial_derivative<^^two_arg, 1, 1>(x, y), 0.0); 
  }
  // Higher order over a DAG with a shared subexpression (a = x*y feeds both
  // parents): f = (xy) sin(xy).
  { 
    double x = 0.7, y = 1.9, p = x*y;
    check("shared d2/dx2", ad::partial_derivative<^^shared, 0, 0>(x, y),
          y*y*(2*std::cos(p) - p*std::sin(p)));
    check("shared d2/dxdy", ad::partial_derivative<^^shared, 0, 1>(x, y),
          std::sin(p) + 3*p*std::cos(p) - p*p*std::sin(p));
    check("mixed commutes", ad::partial_derivative<^^shared, 0, 1>(x, y),
          ad::partial_derivative<^^shared, 1, 0>(x, y)); 
  }

  // Derivatives are usable in constant expressions (poly is pure arithmetic).
  static_assert(ad::forward_derivative<^^poly, 0, double>(4.0) == 2*4.0 + 2);
  static_assert(ad::gradient_reverse<^^poly, double>(4.0)[0]  == 2*4.0 + 2);
  static_assert(ad::partial_derivative<^^poly, 0, 0>(4.0) == 2.0);
  static_assert(ad::partial_derivative<^^poly, 0, 0, 0>(4.0) == 0.0);
  static_assert(ad::partial_derivative<^^two_arg, 0, 1>(0.5, 2.0) == 1.0);

  std::printf(failures ? "\n%d FAILED\n" : "\nALL CHECKS PASSED\n", failures);
  return failures ? 1 : 0;
}
