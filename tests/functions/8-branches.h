#ifndef BRANCHES_H_INCLUDED
#define BRANCHES_H_INCLUDED

// Functions that branch via `if`/`else` (a statement) rather than the
// conditional operator (an expression).

// Both branches reassign the same variable.
inline double abs_via_if(double x) {
  double r = 0.0;
  if (x < 0.0) {
    r = -x;
  } else {
    r = x;
  }
  return r;
}

// No `else`: the untouched branch keeps the pre-if value.
inline double clamp_floor_via_if(double x, double lo) {
  double r = x;
  if (x < lo) {
    r = lo;
  }
  return r;
}

// An `if` with its own init-statement, plus `else if` chaining (nested ifs).
inline double staircase_via_if(double x) {
  double r = 0.0;
  if (double t = x; t < 1.0) {
    r = t * t;
  } else if (t < 2.0) {
    r = 2.0 * t - 1.0;
  } else {
    r = t;
  }
  return r;
}

#endif
