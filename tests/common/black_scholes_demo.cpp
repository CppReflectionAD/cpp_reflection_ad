#include <test_simple_include.hpp>

#include <ad/autograd.h>
#include <ad/forward_derivative.h>

#include "mc_sim/black_scholes.hpp"
#include "mc_sim/black_scholes_reference.hpp"

// The reference values are exact (tools/black_scholes_sympy_value_check.py
// generates them), so the double results are held to 1e-8 of the truth.
namespace ref = black_scholes_reference;

int main() {
  double S = ref::S;
  double K = ref::K;
  double v = ref::v;
  double T = ref::T;

  EXPECT_NEAR_ABS(call_price(S, K, v, T), ref::price, 1e-8);

  // forward mode: double vs float (quantized), results compared side-by-side
  // float has ~7 significant decimal digits, so 1e-4 relative tolerance is
  // safe

  // d/dS
  {
    auto const result_d = ad::forward_derivative<^^call_price, 0>(S, K, v, T);
    static_assert(std::is_same_v<decltype(result_d), const double>,
                  "forward_derivative<0> must return double");
    EXPECT_NEAR_ABS(result_d, ref::d_dS, 1e-8);

    auto const result_f =
        ad::forward_derivative<^^call_price, 0, float>(S, K, v, T);
    static_assert(std::is_same_v<decltype(result_f), const float>,
                  "quantized forward_derivative<0, float> must return float");
    EXPECT_NEAR_ABS(result_f, 0.63904857635498047, 1e-4);

    EXPECT_NEAR_REL(result_d, result_f, 1e-4);
  }

  // d/dK
  {
    auto const result_d = ad::forward_derivative<^^call_price, 1>(S, K, v, T);
    static_assert(std::is_same_v<decltype(result_d), const double>,
                  "forward_derivative<1> must return double");
    EXPECT_NEAR_ABS(result_d, ref::d_dK, 1e-8);

    auto const result_f =
        ad::forward_derivative<^^call_price, 1, float>(S, K, v, T);
    static_assert(std::is_same_v<decltype(result_f), const float>,
                  "quantized forward_derivative<1, float> must return float");
    EXPECT_NEAR_ABS(result_f, -0.54574549198150635, 1e-4);

    EXPECT_NEAR_REL(result_d, result_f, 1e-4);
  }

  // d/dv
  {
    auto const result_d = ad::forward_derivative<^^call_price, 2>(S, K, v, T);
    static_assert(std::is_same_v<decltype(result_d), const double>,
                  "forward_derivative<2> must return double");
    EXPECT_NEAR_ABS(result_d, ref::d_dv, 1e-8);

    auto const result_f =
        ad::forward_derivative<^^call_price, 2, float>(S, K, v, T);
    static_assert(std::is_same_v<decltype(result_f), const float>,
                  "quantized forward_derivative<2, float> must return float");
    EXPECT_NEAR_ABS(result_f, 42.763103485107422, 1e-4);

    EXPECT_NEAR_REL(result_d, result_f, 1e-4);
  }

  // d/dT
  {
    auto const result_d = ad::forward_derivative<^^call_price, 3>(S, K, v, T);
    static_assert(std::is_same_v<decltype(result_d), const double>,
                  "forward_derivative<3> must return double");
    EXPECT_NEAR_ABS(result_d, ref::d_dT, 1e-8);

    auto const result_f =
        ad::forward_derivative<^^call_price, 3, float>(S, K, v, T);
    static_assert(std::is_same_v<decltype(result_f), const float>,
                  "quantized forward_derivative<3, float> must return float");
    EXPECT_NEAR_ABS(result_f, 3.9199509620666504, 1e-4);

    EXPECT_NEAR_REL(result_d, result_f, 1e-4);
  }

  TEST_END;
}
