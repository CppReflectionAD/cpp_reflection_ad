#ifndef COMMA_H_INCLUDED
#define COMMA_H_INCLUDED

#include <cmath>

// The comma operator's left operand has a side effect that is used in
// evaluating the right operand.
inline double assign_then_use(double x, double y) {
  double t;
  return (t = x * y, t * std::sin(t));
}

#endif
