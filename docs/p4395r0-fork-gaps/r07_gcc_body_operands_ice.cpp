// GCC: body_of(f) of a body with a local is_expression; its operands_of hands
// out a raw BLOCK, and is_statement on that ICEs.
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
double f(double x) {
  double v = x * 2;
  return v;
}
constexpr std::meta::info body = std::meta::body_of(^^f);
static_assert(std::meta::is_expression(body), "GCC: the body is an expression");
#if WHICH == 1
static_assert(false, "operands_of(body).size()=" +
                         num(std::meta::operands_of(body).size()));
#elif WHICH == 2
constexpr bool b = std::meta::is_statement(std::meta::operands_of(body)[0]);
#elif WHICH == 3
constexpr bool b = std::meta::is_statement(std::meta::operands_of(body)[1]);
#elif WHICH == 4
constexpr bool b = std::meta::is_statement(std::meta::operands_of(body)[2]);
#endif
