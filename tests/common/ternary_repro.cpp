// ternary_repro.cpp
#include <ad/meta.h>
namespace m = ad::meta;

double relu(double x) { return x > 0.0 ? x : 0.0; }

consteval m::info ret_expr(m::info fn) {
  for (m::info s : m::statements_of(m::body_of(fn)))
    if (m::is_return_statement(s))
      return m::return_operand_of(s);
  throw "no return";
}
consteval m::info strip(m::info e) {
  while (m::is_conversion(e))
    e = m::operands_of(e).front();
  return e;
}

constexpr m::info E = strip(ret_expr(^^relu));

// (1) GCC's ternary reflection -- expected to pass
static_assert(m::is_conditional_operator(E));
static_assert(m::is_operator_expression(strip(m::condition_of(E))) &&
              m::operands_of(strip(m::condition_of(E))).size() == 2);
static_assert(m::is_id_expression(strip(m::true_expression_of(E))));
static_assert(m::is_literal(strip(m::false_expression_of(E))));

#ifdef SHOW_LIBRARY_PATH
// (2) What autograd.h does on GCC when the ternary is skipped:
constexpr auto v =
    m::constant_of(E); // expected: 'x' is not a constant expression
#endif

int main() { return 0; }
