#ifndef LOOPS_H_INCLUDED
#define LOOPS_H_INCLUDED

// Functions with a compile-time-bounded `for` loop.

// A fixed count of additions: s = 5*x.
inline double repeated_add(double x) {
  double s = 0.0;
  for (int i = 0; i < 5; ++i) {
    s = s + x;
  }
  return s;
}

// Counting down with a `-=` step: five iterations (10, 8, 6, 4, 2).
inline double countdown_sum(double x) {
  double s = x;
  for (int i = 10; i > 0; i -= 2) {
    s = s + 1.0;
  }
  return s;
}

// The loop body itself branches -- composes with the guard/Select machinery.
// sum of |x - i| for i in {0, 1, 2}.
inline double sum_of_abs_steps(double x) {
  double s = 0.0;
  for (int i = 0; i < 3; ++i) {
    double t = x - i;
    if (t < 0.0) {
      s = s - t;
    } else {
      s = s + t;
    }
  }
  return s;
}

// Nested loops: s = 9*x.
inline double nested_sum(double x) {
  double s = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      s = s + x;
    }
  }
  return s;
}

#endif
