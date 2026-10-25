// GCC: operands_of a local variable (reached via the declaration statement).
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
constexpr std::meta::info decl =
    std::meta::statements_of(std::meta::body_of(^^f))[0];
constexpr std::meta::info var = std::meta::operands_of(decl)[0];
static_assert(std::meta::is_variable(var));
#if WHICH == 0
static_assert(false, "operands_of(var).size()=" +
                         num(std::meta::operands_of(var).size()));
#elif WHICH == 1
constexpr std::meta::info o = std::meta::operands_of(var)[0];
static_assert(false, std::string("is_variable(o)=") +
                         (std::meta::is_variable(o) ? "1" : "0"));
#elif WHICH == 2
constexpr std::meta::info o = std::meta::operands_of(var)[0];
static_assert(false, std::string("is_statement(o)=") +
                         (std::meta::is_statement(o) ? "1" : "0"));
#elif WHICH == 3
constexpr std::meta::info o = std::meta::operands_of(var)[0];
static_assert(false, std::string("o==var: ") + (o == var ? "1" : "0"));
#elif WHICH == 4
constexpr std::meta::info o = std::meta::operands_of(var)[1];
static_assert(false, std::string("is_statement(o[1])=") +
                         (std::meta::is_statement(o) ? "1" : "0"));
#endif
