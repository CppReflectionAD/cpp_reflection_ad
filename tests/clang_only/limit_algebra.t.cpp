#include "limit_algebra.hpp"

#include <test_simple_include.hpp>

#include "../functions/1-poly.h"
#include "../functions/2-trig.h"
#include "../functions/3-two_arg.h"
#include "../functions/6-piecewise.h"
#include "../tests/mc_sim/black_scholes.hpp"

#include <cmath>

using ad::limits::At;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

inline double fn_inv(double x) { return 1.0 / x; }
inline double fn_inv_shift(double x) { return 1.0 / (1.0 + x); }
inline double fn_sqrt(double x) { return std::sqrt(x); }
inline double fn_log(double x) { return std::log(x); }
inline double fn_exp(double x) { return std::exp(x); }
inline double fn_decay(double x) { return std::exp(-x); }
inline double fn_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double fn_erfc(double x) { return std::erfc(x); }
inline double fn_pole_at_two(double x) { return 1.0 / (x - 2.0); }
inline double fn_abs(double x) { return std::fabs(x); }
inline double fn_recip_sq(double x) { return 1.0 / (x * x); }
inline double fn_step(double x) { return x < 1.0 ? 0.0 : 1.0; }
inline double fn_cancel(double x) { return x - x; }
inline double fn_removable(double x) { return x * x / x; }
inline double fn_ratio(double x) { return x / x; }

consteval bool near(double a, double b, double tol) {
  const double d = a - b;
  return (d < 0.0 ? -d : d) <= tol;
}

static_assert(ad::limits::limit_of<^^fn_sigmoid>(At::plus_infinity()) == 1.0);
static_assert(ad::limits::limit_of<^^fn_sigmoid>(At::minus_infinity()) == 0.0);
static_assert(ad::limits::limit_of<^^fn_decay>(At::plus_infinity()) == 0.0);
static_assert(ad::limits::limit_of<^^fn_inv>(At::plus_infinity()) == 0.0);
static_assert(ad::limits::limit_of<^^fn_inv_shift>(At::plus_infinity()) == 0.0);
static_assert(ad::limits::limit_of<^^fn_erfc>(At::plus_infinity()) == 0.0);
static_assert(ad::limits::limit_of<^^fn_erfc>(At::minus_infinity()) == 2.0);
static_assert(ad::limits::limit_of<^^fn_exp>(At::minus_infinity()) == 0.0);
static_assert(ad::limits::limit_of<^^poly>(At{2.0}) == 8.0);
static_assert(ad::limits::limit_of<^^fn_inv>(At{2.0}) == 0.5);
static_assert(ad::limits::limit_of<^^fn_cancel>(At{3.0}) == 0.0);
static_assert(ad::limits::limit_of<^^fn_ratio>(At{3.0}) == 1.0);
static_assert(ad::limits::limit_of<^^fn_abs>(At{0.0}) == 0.0);
static_assert(ad::limits::limit_of<^^clamp_to>(At{0.5}, 0.0, 1.0) == 0.5);

// The side is carried, not just the value: the sigmoid comes down onto 1 from
// below, and 1/x comes down onto 0 from above.
static_assert(ad::limits::limit_result<^^fn_sigmoid>(At::plus_infinity())
                  .limit.side == ad::limits::Side::Below);
static_assert(ad::limits::limit_result<^^fn_inv>(At::plus_infinity())
                  .limit.side == ad::limits::Side::Above);

// Poles, domain boundaries, one-sided limits.
static_assert(!ad::limits::is_convergent_at<^^fn_inv>(At{0.0}));
static_assert(!ad::limits::is_convergent_at<^^fn_inv>(At::from_right(0.0)));
static_assert(!ad::limits::is_convergent_at<^^fn_recip_sq>(At{0.0}));
static_assert(!ad::limits::is_convergent_at<^^fn_pole_at_two>(At{2.0}));
static_assert(ad::limits::limit_of<^^fn_pole_at_two>(At{2.5}) == 2.0);

static_assert(ad::limits::is_convergent_at<^^fn_sqrt>(At::from_right(0.0)));
static_assert(ad::limits::limit_of<^^fn_sqrt>(At::from_right(0.0)) == 0.0);
static_assert(!ad::limits::is_convergent_at<^^fn_sqrt>(At{0.0}));
static_assert(!ad::limits::is_convergent_at<^^fn_sqrt>(At::from_left(0.0)));
static_assert(ad::limits::limit_of<^^fn_sqrt>(At::from_left(4.0)) == 2.0);
static_assert(!ad::limits::is_convergent_at<^^fn_log>(At::from_right(0.0)));

// Divergence to an infinity
static_assert(!ad::limits::is_convergent_at<^^fn_exp>(At::plus_infinity()));
static_assert(!ad::limits::is_convergent_at<^^poly>(At::plus_infinity()));
static_assert(!ad::limits::is_convergent_at<^^fn_log>(At::plus_infinity()));
static_assert(ad::limits::limit_result<^^fn_log>(At::from_right(0.0))
                  .limit.kind == ad::limits::Value::Kind::MinusInf);

// A jump needs a side.
static_assert(!ad::limits::is_convergent_at<^^fn_step>(At{1.0}));
static_assert(ad::limits::limit_of<^^fn_step>(At::from_left(1.0)) == 0.0);
static_assert(ad::limits::limit_of<^^fn_step>(At::from_right(1.0)) == 1.0);
// A limit away from the jump also works, on both sides.
static_assert(ad::limits::limit_of<^^fn_step>(At{0.0}) == 0.0);
static_assert(ad::limits::limit_of<^^fn_step>(At{2.0}) == 1.0);

static_assert(!ad::limits::is_convergent_at<^^blend>(At{0.5}, 2.0, 3.0));
static_assert(ad::limits::limit_of<^^blend>(At::from_left(0.5), 2.0, 3.0) ==
              5.0);
static_assert(ad::limits::limit_of<^^blend>(At::from_right(0.5), 2.0, 3.0) ==
              6.0);
static_assert(ad::limits::limit_of<^^blend>(1.0, At{2.0}, 3.0) == 6.0);
static_assert(ad::limits::limit_of<^^blend>(0.0, At{2.0}, 3.0) == 5.0);

// The dead side of a decided branch may hold a domain error.
static_assert(ad::limits::is_convergent_at<^^guarded_sqrt>(At{-1.0}));
static_assert(ad::limits::limit_of<^^guarded_sqrt>(At{-1.0}) == 0.0);
static_assert(ad::limits::limit_of<^^guarded_sqrt>(At{4.0}) == 2.0);

// relu is continuous at 0
static_assert(ad::limits::is_convergent_at<^^relu>(At{0.0}));
static_assert(ad::limits::limit_of<^^relu>(At{0.0}) == 0.0);
static_assert(ad::limits::limit_of<^^relu>(At::from_right(0.0)) == 0.0);

// staircase's first seam: x² and 2x-1 both tend to 1 at x = 1.
static_assert(ad::limits::is_convergent_at<^^staircase>(At{1.0}));
static_assert(ad::limits::limit_of<^^staircase>(At{1.0}) == 1.0);
static_assert(ad::limits::limit_of<^^staircase>(At::from_left(1.0)) == 1.0);

// ...and the second seam, where 2x-1 meets exp(x-2)+2 at x = 2. Both sides are
// 3, so this one is continuous too.
static_assert(ad::limits::limit_of<^^staircase>(At{2.0}) == 3.0);

// Away from the seam both engines are fine.
static_assert(ad::limits::limit_of<^^relu>(At{1.0}) == 1.0);
static_assert(ad::limits::limit_of<^^relu>(At{-1.0}) == 0.0);
static_assert(ad::limits::limit_of<^^staircase>(At{0.5}) == 0.25);
static_assert(ad::limits::limit_of<^^staircase>(At{1.5}) == 2.0);

// ---------------------------------------------------------------------------
// Black-Scholes at expiry
// ---------------------------------------------------------------------------
// As T -> 0+ a call is worth its intrinsic value, max(S - K, 0). The price
// itself is undefined at T = 0 -- total_vol is 0 there and d1 divides by it --
// so this is a limit, not an evaluation, and the sides carry it:
//
//   total_vol = 2*sqrt(0+)          -> 0+
//   d1        = log(101/99) / 0+    -> +inf     (a positive over 0+)
//   d2        = d1 - 0+             -> +inf
//   cfd(+inf) = 0.5*erfc(+inf * -0.707) = 0.5*erfc(-inf) -> 1
//   101*1 - 99*1                    -> 2
//
// Every step needs the side: without it log(S/K)/total_vol is "nonzero over 0"
// with no sign, and erfc's argument has no infinity to saturate at.
static_assert(ad::limits::limit_of<^^call_price>(101.0, 99.0, 2.0,
                                                 At::from_right(0.0)) == 2.0);

// Out of the money the same machinery gives the other intrinsic value: the
// log flips sign, so d1 and d2 go to -inf and both cfd's vanish.
static_assert(ad::limits::limit_of<^^call_price>(99.0, 101.0, 2.0,
                                                 At::from_right(0.0)) == 0.0);

// T -> 0 two-sided has no answer: negative T is outside sqrt's domain, so the
// left walk stops at the Sqrt rather than inventing a value.
static_assert(!ad::limits::is_convergent_at<^^call_price>(101.0, 99.0, 2.0,
                                                          At{0.0}));
static_assert(!ad::limits::is_convergent_at<^^call_price>(101.0, 99.0, 2.0,
                                                          At::from_left(0.0)));

// Document questionable behaviour. Should be 0?
static_assert(!ad::limits::is_convergent_at<^^fn_cancel>(At::plus_infinity()));

// inf-inf, 0*inf, inf/inf, 0/0 have no rule, so anything whose limit exists
// only by cancellation reports no limit. Pinned as today's behaviour.
static_assert(!ad::limits::is_convergent_at<^^fn_removable>(At{0.0})); // is 0
static_assert(!ad::limits::is_convergent_at<^^fn_ratio>(At::plus_infinity()));
static_assert(!ad::limits::is_convergent_at<^^two_arg>(At::minus_infinity()));
// sin/cos have no constexpr kernel, so a finite point is None -- the same hole
// the interval engine has, for the same reason.
static_assert(!ad::limits::is_convergent_at<^^trig>(At{2.0}));

int main() {
  // A finite limit point reports no failing node and an exact value.
  {
    constexpr auto r = ad::limits::limit_result<^^poly>(At{2.0});
    EXPECT_TRUE(r.converges);
    EXPECT_EQUAL(r.failing_node, -1);
    EXPECT_EQUAL(r.value, 8.0);
  }

  // A pole is not a failure of the walk: the limit is an infinity, and it is
  // reported as one rather than as a node that gave up.
  {
    constexpr auto r = ad::limits::limit_result<^^fn_inv>(At::from_right(0.0));
    EXPECT_FALSE(r.converges);
    EXPECT_TRUE(r.limit.kind == ad::limits::Value::Kind::PlusInf);
    EXPECT_EQUAL(r.failing_node, -1);
  }
  {
    constexpr auto r = ad::limits::limit_result<^^fn_inv>(At::from_left(0.0));
    EXPECT_TRUE(r.limit.kind == ad::limits::Value::Kind::MinusInf);
  }

  // An indeterminate form names the op it gave up on.
  {
    constexpr auto r = ad::limits::limit_result<^^fn_removable>(At{0.0});
    EXPECT_FALSE(r.converges);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Div);
    EXPECT_TRUE(r.failing_node >= 0);
  }
  {
    constexpr auto r =
        ad::limits::limit_result<^^fn_cancel>(At::plus_infinity());
    EXPECT_FALSE(r.converges);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Sub);
  }

  {
    constexpr auto r = ad::limits::limit_result<^^call_price>(
        101.0, 99.0, 2.0, At::from_right(0.0));
    EXPECT_TRUE(r.converges);
    EXPECT_EQUAL(r.value, 2.0);
    EXPECT_NEAR_ABS(call_price(101.0, 99.0, 2.0, 1e-10), 2.0, 1e-6);
    EXPECT_NEAR_ABS(call_price(99.0, 101.0, 2.0, 1e-10), 0.0, 1e-6);
  }

  TEST_END;
}
