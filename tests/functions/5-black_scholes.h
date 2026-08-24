#ifndef BLACK_SCHOLES_H_INCLUDED
#define BLACK_SCHOLES_H_INCLUDED

#include <cmath>

inline auto call_price(double S, double K, double v, double T) -> double {
  double totalvol = v * std::sqrt(T);
  double d1 = std::log(S / K) / totalvol + totalvol * 0.5;
  double d2 = d1 - totalvol;
  double cfd_d1 = 0.5 * std::erfc(d1 * -0.70710678118654746);
  double cfd_d2 = 0.5 * std::erfc(d2 * -0.70710678118654746);
  return S * cfd_d1 - K * cfd_d2;
}

#endif
