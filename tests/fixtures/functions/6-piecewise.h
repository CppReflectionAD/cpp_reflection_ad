#ifndef PIECEWISE_H_INCLUDED
#define PIECEWISE_H_INCLUDED

#include <cmath>

// Functions whose body branches, via the conditional expression.

// The plainest branch: a linear function on each side of zero.
// f'(x) = 1 for x > 0, 0 for x < 0.
inline double relu(double x) { return x > 0.0 ? x : 0.0; }

// The reason branches are predicated rather than evaluated eagerly. At x < 0
// the true branch would be sqrt of a negative; computing it and then
// discarding it still poisons the reverse sweep, because the adjoint rule for
// sqrt divides by that NaN and `0 / NaN` is NaN, not 0.
// f'(x) = 1/(2*sqrt(x)) for x > 0, 0 for x < 0.
inline double guarded_sqrt(double x) { return x > 0.0 ? std::sqrt(x) : 0.0; }

// A condition on one parameter selecting between two expressions in the
// others; the gradient is that of whichever branch is live.
inline double blend(double flag, double x, double y) {
  return flag > 0.5 ? x * y : x + y;
}

// Nested ternaries, so the inner branches carry a conjoined predicate
// (`And(outer, inner)`) rather than a bare one.
inline double staircase(double x) {
  return x < 1.0 ? x * x : (x < 2.0 ? 2.0 * x - 1.0 : std::exp(x - 2.0) + 2.0);
}

// A branch inside a larger expression, sharing subexpressions with it, so the
// Select is not the root of the DAG.
inline double smooth_step(double x, double k) {
  double t = k * x;
  return (t > 0.0 ? std::sin(t) : t) + t * t;
}

// abs/min/max: continuous, kinked, and recognised as ops rather than desugared
// into branches. d/dx clamp = 1 strictly inside [lo, hi], 0 outside.
// <algorithm>'s std::max/std::min are recognised too (autograd.h registers
// probes for both); fmax/fmin are used here only so this header does not
// depend on <algorithm>.
inline double clamp_to(double x, double lo, double hi) {
  return std::fmin(std::fmax(x, lo), hi);
}

inline double abs_scaled(double x, double s) { return s * std::fabs(x); }

// Short-circuit: at x == 0 the right operand of `&&` must not be evaluated,
// since it divides by x.
inline double safe_recip(double x) {
  return (x != 0.0 && 1.0 / x > 0.5) ? 1.0 / x : 0.0;
}

// `||` takes a different lowering path from `&&` -- its right operand is
// guarded by the *negation* of the left -- so it needs its own coverage.
// Outside [0,1] this is 0; inside, it is x.
inline double window(double x) { return (x < 0.0 || x > 1.0) ? 0.0 : x; }

// The remaining comparison operators. Each maps to its own OpKind in binOp, so
// a mis-mapping (`<=` lowered as `<`) would otherwise go unnoticed.
inline double ramp_le(double x) { return x <= 1.0 ? x * x : 2.0 * x - 1.0; }
inline double ramp_ge(double x) { return x >= 0.0 ? x : -x; }
inline double pick_eq(double x, double y) { return x == 2.0 ? x * y : x + y; }

#endif
