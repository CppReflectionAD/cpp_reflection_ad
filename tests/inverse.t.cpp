#include "is_invertible.hpp"
#include <test_simple_include.hpp>

#include "mc_sim/normal_distribution.hpp"

#include <cmath>

inline double fn_affine(double x) { return 2.0 * x + 3.0; }
inline double fn_exp(double x) { return std::exp(x); }
inline double fn_log(double x) { return std::log(x); }
inline double fn_reciprocal(double x) { return 1.0 / x; }
inline double fn_affine_then_exp(double x) { return std::exp(2.0 * x + 1.0); }
inline double fn_const_div_shift(double x) { return 3.0 / (x + 2.0); }
inline double fn_sum_itself(double x) { return x + x; }
inline double fn_diff_itself(double x) { return x - x; }
inline double fn_sine_custom(double x) { return std::sin(x); }
inline double fn_arcsine_custom(double y) { return std::asin(y); }

using user_sine_pair = ad::inverse_pair<^^fn_sine_custom, fn_arcsine_custom>;

// The inverse metafunction is available only when the checker can prove
// injectivity.
static_assert(ad::is_invertible<^^fn_affine>());
static_assert(ad::is_invertible<^^fn_exp>());
static_assert(ad::is_invertible<^^fn_log>());
static_assert(ad::is_invertible<^^fn_reciprocal>());
static_assert(ad::is_invertible<^^fn_affine_then_exp>());
static_assert(ad::is_invertible<^^fn_const_div_shift>());
static_assert(ad::is_invertible<^^fn_sum_itself>());
static_assert(ad::is_invertible<^^mcsim::CDF>());
static_assert(ad::is_invertible<^^mcsim::CDF_inverse>());
static_assert(ad::is_invertible<^^fn_sine_custom, user_sine_pair>());
static_assert(!ad::is_invertible<^^fn_diff_itself>());

int main() {
  // ad::inverse returns a callable object that is the inverse of the original
  // function.
  {
    constexpr auto inv = ad::inverse<^^fn_affine>{};
    const double input = 4.0;
    const double output = fn_affine(input);
    EXPECT_NEAR_REL(output, 11.0, 1e-12);
    const double recovered = inv(output);
    EXPECT_NEAR_REL(recovered, input, 1e-12);
  }

  {
    constexpr auto inv = ad::inverse<^^fn_exp>{};
    const double input = 1.75;
    const double output = fn_exp(input);
    const double recovered = inv(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    constexpr auto inv = ad::inverse<^^fn_log>{};
    const double input = 2.5;
    const double output = fn_log(input);
    const double recovered = inv(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    constexpr auto inv = ad::inverse<^^fn_reciprocal>{};
    const double input = 4.0;
    const double output = fn_reciprocal(input);
    const double recovered = inv(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    constexpr auto inv = ad::inverse<^^fn_affine_then_exp>{};
    const double input = 0.3;
    const double output = fn_affine_then_exp(input);
    const double recovered = inv(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    constexpr auto inv = ad::inverse<^^fn_const_div_shift>{};
    const double input = 4.0;
    const double output = fn_const_div_shift(input);
    const double recovered = inv(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    constexpr auto inv = ad::inverse<^^fn_sum_itself>{};
    const double input = 0.4;
    const double output = fn_sum_itself(input);
    const double recovered = inv(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  // ad::inverse_of returns the inverse of a function directly, without needing
  // to store it in a variable.
  {
    const double input = 4.0;
    const double output = fn_affine(input);
    const double recovered = ad::inverse_of<^^fn_affine>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    const double input = 1.75;
    const double output = fn_exp(input);
    const double recovered = ad::inverse_of<^^fn_exp>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    const double input = 2.5;
    const double output = fn_log(input);
    const double recovered = ad::inverse_of<^^fn_log>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    const double input = 4.0;
    const double output = fn_reciprocal(input);
    const double recovered = ad::inverse_of<^^fn_reciprocal>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }
  {
    const double input = 0.3;
    const double output = fn_affine_then_exp(input);
    const double recovered = ad::inverse_of<^^fn_affine_then_exp>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    const double input = 4.0;
    const double output = fn_const_div_shift(input);
    const double recovered = ad::inverse_of<^^fn_const_div_shift>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    const double input = 0.4;
    const double output = fn_sum_itself(input);
    const double recovered = ad::inverse_of<^^fn_sum_itself>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  {
    const double input = 0.25;
    const double output = mcsim::CDF(input);
    const double recovered = ad::inverse_of<^^mcsim::CDF>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-7);
  }

  {
    const double input = 0.73;
    const double output = mcsim::CDF_inverse(input);
    const double recovered = ad::inverse_of<^^mcsim::CDF_inverse>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-7);
  }

  {
    const double input = 0.3;
    const double output = fn_sine_custom(input);
    const double recovered =
        ad::inverse_of<^^fn_sine_custom, double, user_sine_pair>(output);
    EXPECT_NEAR_REL(recovered, input, 1e-10);
  }

  TEST_END;
}
