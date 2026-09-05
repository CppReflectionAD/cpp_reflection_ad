#ifndef BLACK_SCHOLES_H_INCLUDED
#define BLACK_SCHOLES_H_INCLUDED

#include <cmath>

inline double total_vol(double v, double T) { return v * std::sqrt(T); }

inline double cfd(double value) {
  return 0.5 * std::erfc(value * -0.70710678118654746);
}

inline double call_price(double S, double K, double v, double T) {
  double totalvol = total_vol(v, T);
  double d1 = std::log(S / K) / totalvol + totalvol * 0.5;
  double d2 = d1 - totalvol;
  return S * cfd(d1) - K * cfd(d2);
}

#endif
