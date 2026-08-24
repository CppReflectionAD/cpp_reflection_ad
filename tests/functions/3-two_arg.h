#ifndef TWO_ARG_H_INCLUDED
#define TWO_ARG_H_INCLUDED

#include <cmath>

inline auto two_arg(double x, double y) -> double {
  return x * y + std::exp(x);
}

#endif
