#ifndef BLACK_SCHOLES_INVERSE_NORMAL_CDF_HPP
#define BLACK_SCHOLES_INVERSE_NORMAL_CDF_HPP

#include <cmath>
#include <limits>

namespace black_scholes {

inline double clamp_open_01(double u) {
  const double eps = std::numeric_limits<double>::epsilon();
  if (u <= eps)
    return eps;
  if (u >= 1.0 - eps)
    return 1.0 - eps;
  return u;
}

inline double normal_cdf(double x) {
  constexpr double sqrt1_2 = 0.70710678118654752440;
  return 0.5 * std::erfc(-x * sqrt1_2);
}

// Approximation due to Peter J. Acklam, adapted to C++.
inline double inverse_normal_cdf(double u) {
  constexpr double a1 = -39.69683028665376;
  constexpr double a2 = 220.9460984245205;
  constexpr double a3 = -275.9285104469687;
  constexpr double a4 = 138.3577518672690;
  constexpr double a5 = -30.66479806614716;
  constexpr double a6 = 2.506628277459239;

  constexpr double b1 = -54.47609879822406;
  constexpr double b2 = 161.5858368580409;
  constexpr double b3 = -155.6989798598866;
  constexpr double b4 = 66.80131188771972;
  constexpr double b5 = -13.28068155288572;

  constexpr double c1 = -0.007784894002430293;
  constexpr double c2 = -0.3223964580411365;
  constexpr double c3 = -2.400758277161838;
  constexpr double c4 = -2.549732539343734;
  constexpr double c5 = 4.374664141464968;
  constexpr double c6 = 2.938163982698783;

  constexpr double d1 = 0.007784695709041462;
  constexpr double d2 = 0.3224671290700398;
  constexpr double d3 = 2.445134137142996;
  constexpr double d4 = 3.754408661907416;

  constexpr double p_low = 0.02425;
  constexpr double p_high = 1.0 - p_low;

  const double p = clamp_open_01(u);

  if (p < p_low) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }

  if (p <= p_high) {
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  }

  const double q = std::sqrt(-2.0 * std::log(1.0 - p));
  return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
         ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

} // namespace black_scholes

#endif // BLACK_SCHOLES_INVERSE_NORMAL_CDF_HPP
