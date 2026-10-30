// GCC: is_statement on an element returned by statements_of(body_of(f)).
#include <cstddef>
#include <meta>
double f(double x) {
  double v = x * 2;
  return v;
}
consteval std::size_t n() {
  return std::meta::statements_of(std::meta::body_of(^^f)).size();
}
static_assert(n() == 2, "two statements");
#if WHICH == 1
static_assert(std::meta::is_declaration_statement(
    std::meta::statements_of(std::meta::body_of(^^f))[0]));
#elif WHICH == 2
static_assert(std::meta::is_statement(
    std::meta::statements_of(std::meta::body_of(^^f))[0]));
#elif WHICH == 3
static_assert(std::meta::is_expression(
    std::meta::statements_of(std::meta::body_of(^^f))[0]));
#elif WHICH == 4
static_assert(std::meta::is_statement(
    std::meta::statements_of(std::meta::body_of(^^f))[1]));
#endif
