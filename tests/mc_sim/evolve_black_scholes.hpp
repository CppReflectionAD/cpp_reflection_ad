#ifndef EVOLVE_BLACK_SCHOLES_HPP_INCLUDED
#define EVOLVE_BLACK_SCHOLES_HPP_INCLUDED

#include "normal_distribution.hpp"

#include <cmath>

// One Black-Scholes step over dt, driven by a single uniform U(0,1) random
// variable.
inline double evolve_black_scholes(double r, double vol, double dt,
                                   double uniform_u) {
  const double z = mcsim::CDF_inverse(uniform_u);
  const double drift = (r - 0.5 * vol * vol) * dt;
  const double diffusion = vol * std::sqrt(dt) * z;
  return std::exp(drift + diffusion);
}

// One Black-Scholes step over dt, driven by a single standard normal random
// variable.
inline double evolve_black_scholes_normal(double r, double vol, double dt,
                                          double normal_z) {
  const double drift = (r - 0.5 * vol * vol) * dt;
  const double diffusion = vol * std::sqrt(dt) * normal_z;
  return std::exp(drift + diffusion);
}

#endif // EVOLVE_BLACK_SCHOLES_HPP_INCLUDED
