// GCC: which statements_of element crashes is_statement?
#include <cstddef>
#include <meta>
#include <string>
consteval std::string num(std::size_t n) {
  std::string s;
  do {
    s.insert(s.begin(), char('0' + n % 10));
    n /= 10;
  } while (n);
  return s;
}
int sink(int);
double multi(double x) {
  double v = x * 2, w = v + 1;
  return v * w;
}
int nested(int) {
  {
    sink(1);
  }
  return 0;
}
int forloop(int n) {
  for (int i = 0; i < n; ++i)
    sink(i);
  return 0;
}
#ifndef FN
#define FN multi
#endif
consteval std::size_t count() {
  return std::meta::statements_of(std::meta::body_of(^^FN)).size();
}
#if WHICH == 0
static_assert(false, "count=" + num(count()));
#else
static_assert(false, std::string("is_statement[") + num(WHICH - 1) + "]=" +
                         (std::meta::is_statement(std::meta::statements_of(
                              std::meta::body_of(^^FN))[WHICH - 1])
                              ? "1"
                              : "0") +
                         " is_expression=" +
                         (std::meta::is_expression(std::meta::statements_of(
                              std::meta::body_of(^^FN))[WHICH - 1])
                              ? "1"
                              : "0"));
#endif
