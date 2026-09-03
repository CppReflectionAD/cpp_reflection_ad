// ad_control_flow_demo.cpp — AD through the conditional expression.
//
// Reflected bodies may now branch: `c ? a : b`, the comparison/logical
// operators that build `c`, and abs/max/min. The derivative taken is that of
// the branch actually taken, which is correct almost everywhere; at a
// switching surface the function is generally not differentiable and the value
// follows the branch convention.
//
// Branches are predicated, so a branch that is not taken is not evaluated —
// see `guarded_sqrt` below, which is the case that distinguishes this from an
// eager select.

#include "../test_simple_include.hpp"

#include "../autograd.h"

#include "functions/6-piecewise.h"

#include <algorithm>
#include <cmath>

namespace {

// Central difference, for checking derivatives away from a switching surface.
template <class F> double fd(F f, double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

// Does the lowered DAG of `Fn` contain this op? Lets a test assert *how* a
// call was lowered, which a derivative check cannot: a helper that is inlined
// rather than recognised as a primitive still yields the right gradient.
template <std::meta::info Fn> consteval bool lowers_to(ad::OpKind op) {
  for (const ad::Node &n : ad::build_nodes<Fn>())
    if (n.op == op)
      return true;
  return false;
}

}  // namespace

// <algorithm>'s max/min, as opposed to <cmath>'s fmax/fmin.
inline double algo_max(double x, double y) { return std::max(x, y); }
inline double algo_min(double x, double y) { return std::min(x, y); }

// They are registered as primitives, so they lower to a single Max/Min node.
// Were they not, they would be inlined as `a < b ? b : a` -- still correct for
// AD, but a Select, which is_continuous_on cannot prove continuous.
static_assert(lowers_to<^^algo_max>(ad::OpKind::Max));
static_assert(lowers_to<^^algo_min>(ad::OpKind::Min));
static_assert(!lowers_to<^^algo_max>(ad::OpKind::Select));
static_assert(!lowers_to<^^algo_min>(ad::OpKind::Select));

// The same check for the other two lowerings this file depends on.
static_assert(lowers_to<^^abs_scaled>(ad::OpKind::Abs));
static_assert(lowers_to<^^relu>(ad::OpKind::Select));

int main() {
  // --- the plainest branch, both sides -------------------------------------
  {
    EXPECT_NEAR_ABS((ad::forward_derivative<^^relu, 0>(2.0)), 1.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^relu, 0>(-2.0)), 0.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^relu>(2.0)[0], 1.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^relu>(-2.0)[0], 0.0, 1e-12);
  }

  // --- predication: the untaken branch is never evaluated ------------------
  // At x < 0 the true branch is sqrt(x). Evaluated eagerly it is NaN, and in
  // reverse mode sqrt's adjoint rule (adj / (2*val)) would turn that into
  // `0 / NaN` == NaN and fold it into the gradient. Predicated, the whole
  // subgraph is skipped and the answer is exactly zero.
  {
    double const g = ad::gradient_reverse<^^guarded_sqrt>(-1.0)[0];
    EXPECT_EQUAL(std::isnan(g), false);
    EXPECT_EQUAL(g, 0.0);

    EXPECT_EQUAL((ad::forward_derivative<^^guarded_sqrt, 0>(-1.0)), 0.0);

    // ...and the live side is still right.
    EXPECT_NEAR_ABS((ad::forward_derivative<^^guarded_sqrt, 0>(4.0)),
                    1.0 / (2.0 * std::sqrt(4.0)), 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^guarded_sqrt>(4.0)[0],
                    1.0 / (2.0 * std::sqrt(4.0)), 1e-12);
  }

  // --- a condition on one parameter, gradient in the others ----------------
  {
    double const x = 1.7, y = 2.3;

    auto const on = ad::gradient_reverse<^^blend>(1.0, x, y);   // x*y
    EXPECT_NEAR_ABS(on[1], y, 1e-12);
    EXPECT_NEAR_ABS(on[2], x, 1e-12);

    auto const off = ad::gradient_reverse<^^blend>(0.0, x, y);  // x+y
    EXPECT_NEAR_ABS(off[1], 1.0, 1e-12);
    EXPECT_NEAR_ABS(off[2], 1.0, 1e-12);

    // The condition itself is piecewise constant, so d/dflag is zero.
    EXPECT_NEAR_ABS(on[0], 0.0, 1e-12);
    EXPECT_NEAR_ABS(off[0], 0.0, 1e-12);
  }

  // --- nested ternaries: each inner branch carries a conjoined predicate ---
  {
    EXPECT_NEAR_ABS((ad::forward_derivative<^^staircase, 0>(0.5)), 2 * 0.5, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^staircase, 0>(1.5)), 2.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^staircase, 0>(3.0)),
                    std::exp(3.0 - 2.0), 1e-12);

    // ...and the AD result agrees with a finite difference of the real
    // function, taken inside each piece. Both sides must involve the
    // differentiator, or this checks nothing about it.
    for (double x : {0.5, 1.5, 3.0}) {
      EXPECT_NEAR_ABS(ad::gradient_reverse<^^staircase>(x)[0],
                      fd(staircase, x), 1e-6);
    }
  }

  // --- a branch embedded in a larger expression ----------------------------
  {
    double const x = 0.9, k = 1.3, t = k * x;
    // d/dx [ sin(kx) + (kx)^2 ] = k cos(kx) + 2k^2 x
    auto const g = ad::gradient_reverse<^^smooth_step>(x, k);
    EXPECT_NEAR_ABS(g[0], k * std::cos(t) + 2 * k * k * x, 1e-10);

    double const xn = -0.9;
    // d/dx [ kx + (kx)^2 ] = k + 2k^2 x
    auto const gn = ad::gradient_reverse<^^smooth_step>(xn, k);
    EXPECT_NEAR_ABS(gn[0], k + 2 * k * k * xn, 1e-10);
  }

  // --- abs / min / max -----------------------------------------------------
  {
    EXPECT_NEAR_ABS((ad::forward_derivative<^^abs_scaled, 0>(3.0, 2.0)), 2.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^abs_scaled, 0>(-3.0, 2.0)), -2.0, 1e-12);
    // d/ds |x| * s = |x|
    EXPECT_NEAR_ABS((ad::forward_derivative<^^abs_scaled, 1>(-3.0, 2.0)), 3.0, 1e-12);

    // clamp: 1 strictly inside the window, 0 outside it.
    EXPECT_NEAR_ABS((ad::forward_derivative<^^clamp_to, 0>(0.5, 0.0, 1.0)), 1.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^clamp_to, 0>(-1.0, 0.0, 1.0)), 0.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^clamp_to, 0>(2.0, 0.0, 1.0)), 0.0, 1e-12);
    // ...and below the window the gradient flows to `lo` instead.
    EXPECT_NEAR_ABS((ad::forward_derivative<^^clamp_to, 1>(-1.0, 0.0, 1.0)), 1.0, 1e-12);
  }

  // --- short-circuit: `&&` must not evaluate 1/x at x == 0 -----------------
  {
    double const g = ad::gradient_reverse<^^safe_recip>(0.0)[0];
    EXPECT_EQUAL(std::isnan(g), false);
    EXPECT_EQUAL(g, 0.0);

    // d/dx 1/x = -1/x^2, on the branch where the guard holds.
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^safe_recip>(1.5)[0],
                    -1.0 / (1.5 * 1.5), 1e-10);
  }

  // --- `||`, whose right operand is guarded by the negation of the left ----
  {
    // Outside the window the value is a constant, so the derivative is 0;
    // inside it is x. At x = -1 the right operand of `||` is not evaluated.
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^window>(0.5)[0], 1.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^window>(-1.0)[0], 0.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^window>(2.0)[0], 0.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^window, 0>(0.5)), 1.0, 1e-12);
  }

  // --- the remaining comparison operators ----------------------------------
  {
    // ramp_le: x^2 for x <= 1, else 2x-1.
    EXPECT_NEAR_ABS((ad::forward_derivative<^^ramp_le, 0>(0.5)), 1.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^ramp_le, 0>(3.0)), 2.0, 1e-12);
    // ramp_ge is |x| written as a branch; note it takes the true side at 0,
    // unlike OpKind::Abs, because `>=` holds there.
    EXPECT_NEAR_ABS((ad::forward_derivative<^^ramp_ge, 0>(2.0)), 1.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^ramp_ge, 0>(-2.0)), -1.0, 1e-12);
    EXPECT_EQUAL((ad::forward_derivative<^^ramp_ge, 0>(0.0)), 1.0);
    // pick_eq: x*y only on exact equality, else x+y.
    auto const on = ad::gradient_reverse<^^pick_eq>(2.0, 5.0);
    EXPECT_NEAR_ABS(on[0], 5.0, 1e-12);
    auto const off = ad::gradient_reverse<^^pick_eq>(3.0, 5.0);
    EXPECT_NEAR_ABS(off[0], 1.0, 1e-12);
  }

  // --- abs/min/max in reverse mode and at second order ---------------------
  // Their forward rules are checked above; the adjoint rules and the
  // Select-building cases in differentiate() are separate code.
  {
    auto const inside = ad::gradient_reverse<^^clamp_to>(0.5, 0.0, 1.0);
    EXPECT_NEAR_ABS(inside[0], 1.0, 1e-12);
    EXPECT_NEAR_ABS(inside[1], 0.0, 1e-12);
    EXPECT_NEAR_ABS(inside[2], 0.0, 1e-12);

    auto const below = ad::gradient_reverse<^^clamp_to>(-1.0, 0.0, 1.0);
    EXPECT_NEAR_ABS(below[0], 0.0, 1e-12);
    EXPECT_NEAR_ABS(below[1], 1.0, 1e-12);   // gradient flows to `lo`

    auto const above = ad::gradient_reverse<^^clamp_to>(2.0, 0.0, 1.0);
    EXPECT_NEAR_ABS(above[0], 0.0, 1e-12);
    EXPECT_NEAR_ABS(above[2], 1.0, 1e-12);   // ...and to `hi`

    auto const g = ad::gradient_reverse<^^abs_scaled>(-3.0, 2.0);
    EXPECT_NEAR_ABS(g[0], -2.0, 1e-12);
    EXPECT_NEAR_ABS(g[1], 3.0, 1e-12);

    // second order: s*|x| is piecewise linear in x, and d2/dx ds = sign(x).
    EXPECT_NEAR_ABS((ad::partial_derivative<^^abs_scaled, 0, 0>(-3.0, 2.0)), 0.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^abs_scaled, 0, 1>(-3.0, 2.0)), -1.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^abs_scaled, 0, 1>(3.0, 2.0)), 1.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^clamp_to, 0, 0>(0.5, 0.0, 1.0)), 0.0, 1e-12);
  }

  // --- <algorithm>'s max/min differentiate like <cmath>'s -------------------
  {
    auto const g = ad::gradient_reverse<^^algo_max>(1.0, 2.0);
    EXPECT_NEAR_ABS(g[0], 0.0, 1e-12);
    EXPECT_NEAR_ABS(g[1], 1.0, 1e-12);
    auto const h = ad::gradient_reverse<^^algo_min>(1.0, 2.0);
    EXPECT_NEAR_ABS(h[0], 1.0, 1e-12);
    EXPECT_NEAR_ABS(h[1], 0.0, 1e-12);
  }

  // --- ramp_le / ramp_ge in the modes their forward checks above skip ------
  {
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^ramp_le>(0.5)[0], 1.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^ramp_le>(3.0)[0], 2.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^ramp_ge>(-2.0)[0], -1.0, 1e-12);
    // x^2 below the switch (f'' = 2), linear above it (f'' = 0).
    EXPECT_NEAR_ABS((ad::partial_derivative<^^ramp_le, 0, 0>(0.5)), 2.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^ramp_le, 0, 0>(3.0)), 0.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^ramp_ge, 0, 0>(-2.0)), 0.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^window, 0, 0>(0.5)), 0.0, 1e-12);
  }

  // --- forward and reverse are independent engines: they must agree --------
  {
    for (double x : {-2.5, -0.3, 0.0, 0.4, 1.0, 3.1}) {
      // Every single-argument branch function, on both sides of its switch
      // and at it. Two engines with separate rule tables reaching the same
      // answer is a much stronger check than either against a hand value.
      EXPECT_NEAR_ABS(ad::gradient_reverse<^^relu>(x)[0],
                      ad::gradient_of<^^relu>(x)[0], 1e-12);
      EXPECT_NEAR_ABS(ad::gradient_reverse<^^guarded_sqrt>(x)[0],
                      ad::gradient_of<^^guarded_sqrt>(x)[0], 1e-12);
      EXPECT_NEAR_ABS(ad::gradient_reverse<^^staircase>(x)[0],
                      ad::gradient_of<^^staircase>(x)[0], 1e-12);
      EXPECT_NEAR_ABS(ad::gradient_reverse<^^window>(x)[0],
                      ad::gradient_of<^^window>(x)[0], 1e-12);
      EXPECT_NEAR_ABS(ad::gradient_reverse<^^ramp_le>(x)[0],
                      ad::gradient_of<^^ramp_le>(x)[0], 1e-12);
      EXPECT_NEAR_ABS(ad::gradient_reverse<^^ramp_ge>(x)[0],
                      ad::gradient_of<^^ramp_ge>(x)[0], 1e-12);
      EXPECT_NEAR_ABS(ad::gradient_reverse<^^safe_recip>(x)[0],
                      ad::gradient_of<^^safe_recip>(x)[0], 1e-12);
    }
    for (double x : {-1.0, 0.0, 0.5, 2.0}) {
      auto const gr = ad::gradient_reverse<^^clamp_to>(x, 0.0, 1.0);
      auto const gf = ad::gradient_of<^^clamp_to>(x, 0.0, 1.0);
      EXPECT_NEAR_ABS(gr[0], gf[0], 1e-12);
      EXPECT_NEAR_ABS(gr[1], gf[1], 1e-12);
      EXPECT_NEAR_ABS(gr[2], gf[2], 1e-12);
      auto const ar = ad::gradient_reverse<^^abs_scaled>(x, 2.0);
      auto const af = ad::gradient_of<^^abs_scaled>(x, 2.0);
      EXPECT_NEAR_ABS(ar[0], af[0], 1e-12);
      EXPECT_NEAR_ABS(ar[1], af[1], 1e-12);
    }
    for (double f : {0.0, 1.0}) {
      auto const gr = ad::gradient_reverse<^^blend>(f, 1.1, 2.2);
      auto const gf = ad::gradient_of<^^blend>(f, 1.1, 2.2);
      EXPECT_NEAR_ABS(gr[0], gf[0], 1e-12);
      EXPECT_NEAR_ABS(gr[1], gf[1], 1e-12);
      EXPECT_NEAR_ABS(gr[2], gf[2], 1e-12);
    }
  }

  // --- higher order through a branch ---------------------------------------
  {
    // relu is linear on each side, so the second derivative is zero on both.
    EXPECT_NEAR_ABS((ad::partial_derivative<^^relu, 0, 0>(2.0)), 0.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^relu, 0, 0>(-2.0)), 0.0, 1e-12);

    // staircase: x^2 below 1 (f'' = 2), exp(x-2) above 2 (f'' = exp(x-2)).
    EXPECT_NEAR_ABS((ad::partial_derivative<^^staircase, 0, 0>(0.5)), 2.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^staircase, 0, 0>(1.5)), 0.0, 1e-12);
    EXPECT_NEAR_ABS((ad::partial_derivative<^^staircase, 0, 0>(3.0)),
                    std::exp(1.0), 1e-10);

    // The predication survives differentiation: the second derivative of the
    // guarded sqrt is still finite on the dead side.
    double const d2 = (ad::partial_derivative<^^guarded_sqrt, 0, 0>(-1.0));
    EXPECT_EQUAL(std::isnan(d2), false);
    EXPECT_EQUAL(d2, 0.0);
    // d2/dx2 sqrt(x) = -1/(4 x^(3/2))
    EXPECT_NEAR_ABS((ad::partial_derivative<^^guarded_sqrt, 0, 0>(4.0)),
                    -1.0 / (4.0 * std::pow(4.0, 1.5)), 1e-10);
  }

  // --- boundary convention, pinned so a change to it is a test failure -----
  {
    // `x > 0 ? x : 0` takes the false branch at exactly zero.
    EXPECT_EQUAL((ad::forward_derivative<^^relu, 0>(0.0)), 0.0);
    // OpKind::Abs tests `val < 0`, so d|x|/dx at exactly zero is +1.
    EXPECT_EQUAL((ad::forward_derivative<^^abs_scaled, 0>(0.0, 2.0)), 2.0);
    EXPECT_EQUAL(ad::gradient_reverse<^^abs_scaled>(0.0, 2.0)[0], 2.0);
    // max(a,b) is `a < b ? b : a`, so a tie yields the first operand: the
    // gradient flows to x, not to lo.
    EXPECT_EQUAL((ad::forward_derivative<^^clamp_to, 0>(0.0, 0.0, 1.0)), 1.0);
    EXPECT_EQUAL((ad::forward_derivative<^^clamp_to, 1>(0.0, 0.0, 1.0)), 0.0);
  }

  // --- derivatives through a branch are usable in constant expressions -----
  static_assert((ad::forward_derivative<^^relu, 0, double>(2.0)) == 1.0);
  static_assert((ad::forward_derivative<^^relu, 0, double>(-2.0)) == 0.0);
  static_assert((ad::gradient_reverse<^^relu, double>(2.0))[0] == 1.0);
  static_assert((ad::partial_derivative<^^relu, 0, 0>(2.0)) == 0.0);
  static_assert((ad::gradient_reverse<^^blend, double>(1.0, 3.0, 5.0))[1] == 5.0);
  static_assert((ad::gradient_reverse<^^blend, double>(0.0, 3.0, 5.0))[1] == 1.0);

  TEST_END;
}
