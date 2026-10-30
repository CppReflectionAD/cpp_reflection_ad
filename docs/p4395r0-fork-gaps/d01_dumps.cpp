#include "dump.h"
#include <type_traits>
int sink(int);
template <typename T> int doSomething(T) {
  if constexpr (std::is_arithmetic_v<T>) {
    sink(1);
  } else {
    sink(2);
    sink(3);
  }
  return 0;
}
int kinds_fn(int n) {
  int s = 0;
  {
    sink(1);
    sink(2);
  }
  for (int i = 0; i < n; ++i)
    sink(i);
  sink(3);
  return s;
}
double multi(double x) {
  double v = x * 2, w = v + 1;
  return v * w;
}
struct P {
  double a, b;
};
double bind(P p) {
  auto [a, b] = p;
  using D = double;
  return D(a) + b;
}
int max(int x, int y) { return (x > y) ? x : y; }
struct S {
  double v;
};
S operator+(S, S);
S ovl(S a, S b) { return a + b; }
struct Particle {
  double x, y, z;
};
void update(Particle &p) { p.x += p.y * 0.5; }
void incs(int &i) {
  ++i;
  i++;
}
double g(double);
double callg(double x) { return g(x); }

#if WHICH == 1
DUMP(doSomething<int>, 4);
#endif
#if WHICH == 2
DUMP(kinds_fn, 3);
#endif
#if WHICH == 3
DUMP(multi, 3);
#endif
#if WHICH == 4
DUMP(bind, 3);
#endif
#if WHICH == 5
DUMP(max, 5);
#endif
#if WHICH == 6
DUMP(ovl, 4);
#endif
#if WHICH == 7
DUMP(update, 5);
#endif
#if WHICH == 8
DUMP(incs, 4);
#endif
#if WHICH == 9
DUMP(callg, 4);
#endif
