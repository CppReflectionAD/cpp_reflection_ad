// ad_scalar_demo.cpp — scalar AD on reflected functions: forward mode, reverse
// mode, and activity analysis. Pick an ordinary function; its derivative is
// generated at compile time and spliced as inlined arithmetic (no DSL, no
// tape).
//
// Build/run: `make run-scalar`.

#include "test_simple_include.hpp"

#include "autograd.h"

#include "functions/1-poly.h"
#include "functions/2-trig.h"
#include "functions/3-two_arg.h"
#include "functions/4-shared_intemediate.h"

#include <cmath>
#include <cstdio>

int main() {
  // forward mode (one directional derivative)
  {
    auto const result = ad::forward_derivative<^^poly, 0>(3.0);
    EXPECT_NEAR_ABS(result, 2. * 3.0 + 2., 1e-8);
  }

  {
    double const x = 1.3;
    auto const result = ad::forward_derivative<^^trig, 0>(x);
    EXPECT_NEAR_ABS(result, 2 * x * std::cos(x * x), 1e-8);
  }

  // reverse mode (whole gradient in one pass)
  {
    double x = 0.7, y = 1.9, p = x * y;
    auto g = ad::gradient_reverse<^^shared>(x, y); // shared 'a' handled once

    EXPECT_NEAR_ABS(g[0], y * std::sin(p) + p * std::cos(p) * y, 1e-8);
    EXPECT_NEAR_ABS(g[1], x * std::sin(p) + p * std::cos(p) * x, 1e-8);
  }

  // reverse == forward gradient
  {
    double x = 0.5, y = 2.0;
    auto gr = ad::gradient_reverse<^^two_arg>(x, y);
    auto gf = ad::gradient_of<^^two_arg>(x, y);

    EXPECT_NEAR_ABS(gr[0], gf[0], 1e-8);
    EXPECT_NEAR_ABS(gr[1], gf[1], 1e-8);
  }

  // Activity analysis: d/dy of (x*y + exp(x)) is just x — the exp term is
  // pruned (no exp call, no x*0); verified as code quality in `make run-bench`.
  {
    auto const result = ad::forward_derivative<^^two_arg, 1>(0.5, 2.0);
    EXPECT_NEAR_ABS(result, 0.5, 1e-8);
  }

  // Derivatives are usable in constant expressions (poly is pure arithmetic).
  static_assert(ad::forward_derivative<^^poly, 0, double>(4.0) == 2 * 4.0 + 2);
  static_assert(ad::gradient_reverse<^^poly, double>(4.0)[0] == 2 * 4.0 + 2);

  TEST_END;
}
