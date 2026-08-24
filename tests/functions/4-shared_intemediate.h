#ifndef SHARED_H_INCLUDED
#define SHARED_H_INCLUDED

#include <cmath>

// multi-assignment with a shared intermediate (a DAG, not a tree):
inline auto shared(double x, double y) -> double {
  double a = -x * -y;      // a is used twice. Test - unary operator too.
  double b = +std::sin(a); // test + unary operator here too
  return a * b;            // f = (xy) sin(xy)
}

#endif
