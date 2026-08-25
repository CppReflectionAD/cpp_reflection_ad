#include "test_simple_include.hpp"

#include "autograd.h"
#include "functions/5-black_scholes.h"

int main() {
  double S = 104.25;
  double K = 98.5;
  double v = 0.22;
  double T = 1.20;

  // forward mode (one directional derivative)
  {
    auto const result = ad::forward_derivative<^^call_price, 0>(S, K, v, T);
    EXPECT_NEAR_ABS(result, 0.63904872167822369, 1e-8);
  }

  {
    auto const result = ad::forward_derivative<^^call_price, 1>(S, K, v, T);
    EXPECT_NEAR_ABS(result, -0.54574545575331679, 1e-8);
  }

  {
    auto const result = ad::forward_derivative<^^call_price, 2>(S, K, v, T);
    EXPECT_NEAR_ABS(result, 42.763099555399286, 1e-8);
  }

  {
    auto const result = ad::forward_derivative<^^call_price, 3>(S, K, v, T);
    EXPECT_NEAR_ABS(result, 3.919950792578268, 1e-8);
  }

  TEST_END;
}
