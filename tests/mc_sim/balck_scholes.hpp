#ifndef BALCK_SCHOLES_HPP_INCLUDED
#define BALCK_SCHOLES_HPP_INCLUDED

#include "normal_distribution.hpp"
#include <cmath>

inline double total_vol(double v, double T) { return v * std::sqrt(T); }

inline double call_price(double S, double K, double v, double T) {
  double totalvol = total_vol(v, T);
  double d1 = std::log(S / K) / totalvol + totalvol * 0.5;
  double d2 = d1 - totalvol;
  return S * mcsim::CDF(d1) - K * mcsim::CDF(d2);
}

inline double put_price(double S, double K, double v, double T) {
  double totalvol = total_vol(v, T);
  double d1 = std::log(S / K) / totalvol + totalvol * 0.5;
  double d2 = d1 - totalvol;
  return K * mcsim::CDF(-d2) - S * mcsim::CDF(-d1);
}

#endif
