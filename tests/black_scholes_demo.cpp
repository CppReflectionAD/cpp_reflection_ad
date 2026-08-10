#include "autograd.h"
#include "black_scholes.h"

#include <cmath>
#include <cstdio>
#include <random>

// // Ordinary functions, differentiated by reflection without modification.
// constexpr double poly(double x) { return x * x + 2.0 * x; } // f' = 2x+2
// constexpr double trig(double x) { return std::sin(x * x); } // f' = 2x
// cos(x^2) constexpr double two_arg(double x, double y) {
//   return x * y + std::exp(x);
// } // ∇=[y+e^x, x]
// // multi-assignment with a shared intermediate (a DAG, not a tree):
// constexpr double shared(double x, double y) {
//   double a = x * y; // used twice
//   double b = std::sin(a);
//   return a * b; // f = (xy) sin(xy)
// }

static int failures = 0;
static void check(const char *name, double got, double want) {
  bool ok = std::abs(got - want) < 1e-6 * (1.0 + std::abs(want));
  std::printf("  %-22s %+.6f (want %+.6f) %s\n", name, got, want,
              ok ? "ok" : "FAIL");
  failures += !ok;
}

int main() {

  //   std::mt19937 generator(123);
  //   std::uniform_real_distribution<double> stock_distr(90, 110.0),
  //       vol_distr(0.05, 0.3), time_distr(0.5, 1.5);

  //   double S = stock_distr(generator);
  //   double K = stock_distr(generator);
  //   double v = vol_distr(generator);
  //   double T = time_distr(generator);

  double S = 104.25910643160562;
  double K = 98.569418500578678;
  double v = 0.22272121285943147;
  double T = 1.2191503088807956;

  std::printf("forward mode  (one directional derivative):\n");
  check("call_price'(0)", ad::forward_derivative<^^call_price, 0>(S, K, v, T),
        0.63726496508722587043);

  check("call_price'(1)", ad::forward_derivative<^^call_price, 1>(S, K, v, T),
        -0.54190720509687984535);

  check("call_price'(2)", ad::forward_derivative<^^call_price, 2>(S, K, v, T),
        43.179329512582194174);

  check("call_price'(3)", ad::forward_derivative<^^call_price, 3>(S, K, v, T),
        3.9441209871520704033);

  return failures ? 1 : 0;
}
