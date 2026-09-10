#ifndef EVOLVE_BLACK_SCHOLES_HPP_INCLUDED
#define EVOLVE_BLACK_SCHOLES_HPP_INCLUDED

#include "normal_distribution.hpp"

#include <cmath>

// One Black-Scholes step over dt, driven by a single uniform U(0,1) random
// variable.
inline double evolve_black_scholes(double spot, double r, double vol, double dt,
                                   double uniform_u) {
  const double z = mcsim::CDF_inverse(uniform_u);
  const double drift = (r - 0.5 * vol * vol) * dt;
  const double diffusion = vol * std::sqrt(dt) * z;
  return spot * std::exp(drift + diffusion);
}

#endif // EVOLVE_BLACK_SCHOLES_HPP_INCLUDED
