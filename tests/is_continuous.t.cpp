#include "is_continuous.hpp"
#include "test_simple_include.hpp"

#include "functions/1-poly.h"
#include "functions/2-trig.h"
#include "functions/3-two_arg.h"
#include "functions/4-shared_intemediate.h"
#include "functions/5-black_scholes.h"
#include "functions/6-piecewise.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Helper functions for negative tests (functions with genuine discontinuities)
// ---------------------------------------------------------------------------

// log(x): discontinuous on any interval containing x <= 0.
inline double fn_log(double x) { return std::log(x); }

// 1/x: discontinuous at x = 0.
inline double fn_inv(double x) { return 1.0 / x; }

// sqrt(x): discontinuous (not real-valued) for x < 0.
inline double fn_sqrt_neg(double x) { return std::sqrt(x); }

// log(x - y): discontinuous when the denominator x - y <= 0.
inline double fn_log_diff(double x, double y) { return std::log(x - y); }

// A branch whose dead side contains a domain error. On a box where the
// condition is decidable, only the live side is checked.
inline double branch_log(double x) { return x > 0.0 ? std::log(x) : 0.0; }

// |x| is continuous across zero (it is merely non-differentiable there), so
// recognising Abs as an op rather than desugaring it into a branch is what
// keeps this from being a false alarm.
inline double fn_abs(double x) { return std::fabs(x); }

// A branch that is decidable on the box, either way.
inline double fn_clamped(double x) { return std::fmin(std::fmax(x, 0.0), 1.0); }

// A domain error under an *undecidable* guard, whose meaningless range is then
// fed through fmin. Nothing may "decide" the outer branch from that range: the
// function really does jump at x = 0 (h(1e-9) == 1, h(-1e-9) == 0). This is a
// regression test — reporting it continuous would break the header's promise
// that a `true` answer is a guarantee.
inline double fn_laundered(double x) {
  return (x > 0.0 && std::fmin(2.0, std::log(x)) < 1.0) ? 1.0 : 0.0;
}

// ---------------------------------------------------------------------------
// Compile-time assertions (static_assert): these are the primary tests.
// They fail at compile time if the implementation is wrong.
// ---------------------------------------------------------------------------

// --- poly(x) = x² + 2x  — entire, continuous everywhere ---
static_assert(ad::is_continuous_on<^^poly>(ad::Interval{-100.0, 100.0}));

static_assert(ad::is_continuous_on<^^poly>(ad::Interval{0.0, 1.0}));

// --- trig(x) = sin(x²)  — entire, continuous everywhere ---
static_assert(ad::is_continuous_on<^^trig>(ad::Interval{-10.0, 10.0}));

// --- branches -------------------------------------------------------------
// Condition decidably true on the box: only the live (log) side is checked,
// and it is fine there.
static_assert(ad::is_continuous_on<^^branch_log>(ad::Interval{1.0, 5.0}));

// Condition decidably false: the log on the dead side is never reached, so a
// domain error there is not the function's problem.
static_assert(ad::is_continuous_on<^^branch_log>(ad::Interval{-2.0, -1.0}));

// Condition undecidable across the box: the function may jump at x = 0.
static_assert(!ad::is_continuous_on<^^branch_log>(ad::Interval{-1.0, 1.0}));

// Same shape, but the two sides happen to meet at zero. The checker reports it
// as discontinuous anyway — sound, and incomplete in the documented direction.
static_assert(!ad::is_continuous_on<^^guarded_sqrt>(ad::Interval{-5.0, 5.0}));
static_assert(ad::is_continuous_on<^^guarded_sqrt>(ad::Interval{1.0, 5.0}));

// abs/min/max are kinks, not jumps: continuous across the switching point.
static_assert(ad::is_continuous_on<^^fn_abs>(ad::Interval{-3.0, 3.0}));
static_assert(ad::is_continuous_on<^^fn_clamped>(ad::Interval{-3.0, 3.0}));

// A meaningless range must not be launderable back into a decision by an op
// that drops NaN (std::min(2.0, NaN) == 2.0).
static_assert(!ad::is_continuous_on<^^fn_laundered>(ad::Interval{-1.0, 1.0}));

// --- the remaining Tri helpers -------------------------------------------
// `||` (tri_or), decided both ways: inside the window both operands are
// provably false, so the branch is settled; a box straddling an edge is not.
static_assert(ad::is_continuous_on<^^window>(ad::Interval{0.2, 0.8}));
static_assert(!ad::is_continuous_on<^^window>(ad::Interval{-2.0, 2.0}));

// `<=` (tri_le) and `>=`, decidable on a box clear of the switching point.
static_assert(ad::is_continuous_on<^^ramp_le>(ad::Interval{2.0, 3.0}));
static_assert(!ad::is_continuous_on<^^ramp_le>(ad::Interval{0.0, 2.0}));
static_assert(ad::is_continuous_on<^^ramp_ge>(ad::Interval{1.0, 3.0}));

// `==` (tri_eq) is decidable only between point or disjoint intervals, so an
// equality-guarded branch over any wider box is reported.
static_assert(!ad::is_continuous_on<^^pick_eq>(ad::Interval{1.0, 3.0},
                                               ad::Interval{0.0, 1.0}));
static_assert(ad::is_continuous_on<^^pick_eq>(ad::Interval{5.0, 6.0},
                                              ad::Interval{0.0, 1.0}));

// --- the same functions the AD tests use, checked from this side ----------
static_assert(ad::is_continuous_on<^^relu>(ad::Interval{1.0, 5.0}));
static_assert(ad::is_continuous_on<^^relu>(ad::Interval{-5.0, -1.0}));
// relu really is continuous at 0, but proving that needs the two branches to
// be compared on the switching surface, which interval arithmetic does not do.
// Pinned as a known-incomplete case, not as desired behaviour.
static_assert(!ad::is_continuous_on<^^relu>(ad::Interval{-1.0, 1.0}));

// A three-parameter box, which the one-argument fn_clamped above cannot reach:
// each bound gets its own interval.
static_assert(ad::is_continuous_on<^^clamp_to>(ad::Interval{-3.0, 3.0},
                                               ad::Interval{0.0, 0.0},
                                               ad::Interval{1.0, 1.0}));

// A branch nested inside a larger expression. The boxes below are all at least
// 2π wide in `t = k*x`, which is not incidental: narrower intervals through a
// Sin/Cos node do not compile today. See the LIMITATION on sin_range in
// is_continuous.hpp.
//   t in [7,20]: the branch is live, and sin_range takes its wide-interval path
static_assert(ad::is_continuous_on<^^smooth_step>(ad::Interval{7.0, 20.0},
                                                  ad::Interval{1.0, 1.0}));
//   t in [-20,-7]: the sin branch is dead and never evaluated at all
static_assert(ad::is_continuous_on<^^smooth_step>(ad::Interval{-20.0, -7.0},
                                                  ad::Interval{1.0, 1.0}));
//   t straddling 0: undecidable branch, so reported as a jump
static_assert(!ad::is_continuous_on<^^smooth_step>(ad::Interval{-20.0, 20.0},
                                                   ad::Interval{1.0, 1.0}));

// --- two_arg(x,y) = x*y + exp(x)  — entire ---
static_assert(ad::is_continuous_on<^^two_arg>(ad::Interval{-5.0, 5.0},
                                              ad::Interval{-5.0, 5.0}));

// --- shared(x,y) = (xy)·sin(xy)  — entire ---
static_assert(ad::is_continuous_on<^^shared>(ad::Interval{-3.0, 3.0},
                                             ad::Interval{-3.0, 3.0}));

// --- call_price: continuous on the standard financial domain ---
static_assert(ad::is_continuous_on<^^call_price>(ad::Interval{90.0, 110.0}, // S
                                                 ad::Interval{90.0, 110.0}, // K
                                                 ad::Interval{0.05, 0.30},  // v
                                                 ad::Interval{0.50,
                                                              1.50})); // T

// --- call_price: also continuous on a wider domain (S and K far from 0) ---
static_assert(ad::is_continuous_on<^^call_price>(ad::Interval{1.0, 1000.0},
                                                 ad::Interval{1.0, 1000.0},
                                                 ad::Interval{0.01, 2.0},
                                                 ad::Interval{0.01, 30.0}));

// --- fn_log(x): continuous only when x > 0 ---
static_assert(ad::is_continuous_on<^^fn_log>(ad::Interval{0.01, 10.0}));
static_assert(!ad::is_continuous_on<^^fn_log>(ad::Interval{-1.0, 10.0}));
static_assert(!ad::is_continuous_on<^^fn_log>(ad::Interval{-5.0, 0.0}));

// --- fn_inv(x) = 1/x: continuous away from 0 ---
static_assert(ad::is_continuous_on<^^fn_inv>(ad::Interval{0.1, 10.0}));
static_assert(ad::is_continuous_on<^^fn_inv>(ad::Interval{-5.0, -0.1}));
static_assert(!ad::is_continuous_on<^^fn_inv>(ad::Interval{-1.0, 1.0}));
static_assert(!ad::is_continuous_on<^^fn_inv>(ad::Interval{0.0, 1.0}));

// --- fn_sqrt_neg(x): continuous only when x >= 0 ---
static_assert(ad::is_continuous_on<^^fn_sqrt_neg>(ad::Interval{0.0, 10.0}));
static_assert(!ad::is_continuous_on<^^fn_sqrt_neg>(ad::Interval{-1.0, 10.0}));

// --- fn_log_diff(x,y) = log(x - y): continuous when x > y everywhere ---
static_assert(ad::is_continuous_on<^^fn_log_diff>(
    ad::Interval{5.0, 10.0}, ad::Interval{0.0, 3.0})); // x - y >= 2.0 > 0

static_assert(!ad::is_continuous_on<^^fn_log_diff>(
    ad::Interval{0.0, 5.0}, ad::Interval{0.0, 5.0})); // x - y can be <= 0

// ---------------------------------------------------------------------------
// Runtime tests: verify that continuity_result() gives correct detail.
// ---------------------------------------------------------------------------

int main() {
  // A passing check returns the right fields.
  {
    constexpr auto r = ad::continuity_result<^^call_price>(
        ad::Interval{90.0, 110.0}, ad::Interval{90.0, 110.0},
        ad::Interval{0.05, 0.30}, ad::Interval{0.50, 1.50});
    EXPECT_TRUE(r.continuous);
    EXPECT_EQUAL(r.failing_node, -1);
  }

  // A failing check on log identifies the right op.
  {
    constexpr auto r =
        ad::continuity_result<^^fn_log>(ad::Interval{-1.0, 10.0});
    EXPECT_FALSE(r.continuous);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Log);
  }

  // A failing check on 1/x identifies Div.
  {
    constexpr auto r = ad::continuity_result<^^fn_inv>(ad::Interval{-1.0, 1.0});
    EXPECT_FALSE(r.continuous);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Div);
  }

  // A failing check on sqrt identifies Sqrt.
  {
    constexpr auto r =
        ad::continuity_result<^^fn_sqrt_neg>(ad::Interval{-1.0, 10.0});
    EXPECT_FALSE(r.continuous);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Sqrt);
  }

  // poly is continuous — no failing node.
  {
    constexpr auto r =
        ad::continuity_result<^^poly>(ad::Interval{-100.0, 100.0});
    EXPECT_TRUE(r.continuous);
  }

  // An undecidable branch is reported against the Select, not against whatever
  // happens to sit inside the branch.
  {
    constexpr auto r =
        ad::continuity_result<^^guarded_sqrt>(ad::Interval{-5.0, 5.0});
    EXPECT_FALSE(r.continuous);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Select);
  }

  // The dead branch of a decidable condition holds a domain error that must
  // not fire: on x < 0 the log is never reached.
  {
    constexpr auto r =
        ad::continuity_result<^^branch_log>(ad::Interval{-2.0, -1.0});
    EXPECT_TRUE(r.continuous);
    EXPECT_EQUAL(r.failing_node, -1);
  }

  TEST_END;
}
