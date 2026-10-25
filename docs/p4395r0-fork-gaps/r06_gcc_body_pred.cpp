// GCC: predicates on body_of(f) itself, for a body with a local.
#include <meta>
#include <vector>
double f(double x) {
  double v = x * 2;
  return v;
}
#if WHICH == 1
constexpr bool b = std::meta::is_statement(std::meta::body_of(^^f));
#elif WHICH == 2
constexpr bool b = std::meta::is_expression(std::meta::body_of(^^f));
#elif WHICH == 3
consteval int walk() {
  int n = 0;
  for (std::meta::info c : std::meta::statements_of(std::meta::body_of(^^f)))
    n += std::meta::is_expression(c);
  return n;
}
static_assert(walk() == 2);
#elif WHICH == 4
consteval int walk() {
  int n = 0;
  std::meta::info b = std::meta::body_of(^^f);
  n += std::meta::is_expression(b);
  return n;
}
static_assert(walk() == 0);
#endif
