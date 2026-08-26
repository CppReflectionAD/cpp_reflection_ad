#include "is_continuous.hpp"
#include "test_simple_include.hpp"

#include "functions/1-poly.h"
#include "functions/2-trig.h"
#include "functions/3-two_arg.h"
#include "functions/4-shared_intemediate.h"
#include "functions/5-black_scholes.h"

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

// ---------------------------------------------------------------------------
// Compile-time assertions (static_assert): these are the primary tests.
// They fail at compile time if the implementation is wrong.
// ---------------------------------------------------------------------------

// --- poly(x) = x² + 2x  — entire, continuous everywhere ---
static_assert(ad::is_continuous_on<^^poly>(ad::Interval{-100.0, 100.0}));

static_assert(ad::is_continuous_on<^^poly>(ad::Interval{0.0, 1.0}));

// --- trig(x) = sin(x²)  — entire, continuous everywhere ---
static_assert(ad::is_continuous_on<^^trig>(ad::Interval{-10.0, 10.0}));

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

  TEST_END;
}
