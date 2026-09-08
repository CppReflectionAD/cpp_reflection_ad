#include "is_invertible.hpp"
#include <test_simple_include.hpp>

#include "mc_sim/normal_distribution.hpp"

#include <cmath>

// ---------------------------------------------------------------------------
// Functions used by invertibility tests
// ---------------------------------------------------------------------------

inline double fn_affine(double x) { return 2.0 * x + 3.0; }
inline double fn_square(double x) { return x * x; }
inline double fn_exp(double x) { return std::exp(x); }
inline double fn_log(double x) { return std::log(x); }
inline double fn_log_exp(double x) { return std::log(std::exp(x)); }
inline double fn_constant(double x) { return 0.0 * x + 1.0; }
inline double fn_reciprocal(double x) { return 1.0 / x; }
inline double fn_sine(double x) { return std::sin(x); }
inline double fn_arcsine(double y) { return std::asin(y); }
inline double fn_two_arg(double x, double y) { return x + y; }

using user_sine_pair = ad::inverse_pair<^^fn_sine, ^^fn_arcsine>;

// ---------------------------------------------------------------------------
// Compile-time assertions (primary tests)
// ---------------------------------------------------------------------------

// Affine with nonzero slope: invertible on R.
static_assert(ad::is_invertible<^^fn_affine>());

// x^2 is not injective on R.
static_assert(!ad::is_invertible<^^fn_square>());

// exp and log are injective on their natural domains.
static_assert(ad::is_invertible<^^fn_exp>());
static_assert(ad::is_invertible<^^fn_log>());

// Composition of injective unary ops remains injective.
static_assert(ad::is_invertible<^^fn_log_exp>());

// Constant map is not invertible.
static_assert(!ad::is_invertible<^^fn_constant>());

// Reciprocal is injective on its largest natural domain (R \ {0}).
static_assert(ad::is_invertible<^^fn_reciprocal>());

// Trig is currently unsupported by the checker, so reported non-invertible.
static_assert(!ad::is_invertible<^^fn_sine>());

// Multivariate functions are currently out of scope for this checker.
static_assert(!ad::is_invertible<^^fn_two_arg>());

// Registered special-case inverse pair for normal CDF.
static_assert(ad::is_invertible<^^mcsim::CDF>());
static_assert(ad::is_invertible<^^mcsim::CDF_inverse>());

// User can inject custom inverse pairs directly via variadic template args.
static_assert(ad::is_invertible<^^fn_sine, user_sine_pair>());

int main() {
  {
    constexpr auto r = ad::invertibility_result<^^fn_affine>();
    EXPECT_TRUE(r.invertible);
    EXPECT_EQUAL(r.failing_node, -1);
  }

  {
    constexpr auto r = ad::invertibility_result<^^fn_square>();
    EXPECT_FALSE(r.invertible);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Mul);
  }

  {
    constexpr auto r = ad::invertibility_result<^^fn_two_arg>();
    EXPECT_FALSE(r.invertible);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Input);
  }

  {
    constexpr auto r = ad::invertibility_result<^^mcsim::CDF>();
    EXPECT_TRUE(r.invertible);
    EXPECT_EQUAL(r.failing_node, -1);
  }

  {
    constexpr auto r = ad::invertibility_result<^^mcsim::CDF_inverse>();
    EXPECT_TRUE(r.invertible);
    EXPECT_EQUAL(r.failing_node, -1);
  }

  {
    constexpr auto r = ad::invertibility_result<^^fn_sine, user_sine_pair>();
    EXPECT_TRUE(r.invertible);
    EXPECT_EQUAL(r.failing_node, -1);
  }

  TEST_END;
}
