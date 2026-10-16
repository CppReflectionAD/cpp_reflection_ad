// GCC reflection gap: `&&`, `||` and `!` are not classified as operators.
//
// For a comparison such as `x > 0.0`, both compilers report
// is_binary_operator == true. For the logical operators, clang reports
// is_binary_operator / is_unary_operator == true, but GCC reports false --
// even though GCC's expression_operator_of() and operands_of() already
// handle these expressions correctly.
//
// This file checks what works on both compilers. The classification asserts
// that fail on GCC are in tests/known_gap/gcc/logical_operator_reflection.cpp
// (which expects them to fail) and tests/clang_only/
// logical_operator_reflection.cpp (which expects them to pass).
//
// Build:
//   g++     -std=c++2c -freflection -fsyntax-only
//   gcc_logical_operator_reflection.cpp
//   clang++ -std=c++26 -freflection-latest -stdlib=libc++ -fsyntax-only \
//           gcc_logical_operator_reflection.cpp
//   (on macOS, also pass -isysroot "$(xcrun --show-sdk-path)" to clang)
//
// Expected: compiles cleanly on both compilers.

#include <meta>

namespace m = std::meta;

bool f_cmp(double x) { return x > 0.0; }
bool f_or(double x) { return (x > 0.0) || (x > 1.0); }
bool f_and(double x) { return (x > 0.0) && (x > 1.0); }
bool f_not(double x) { return !(x > 0.0); }

// The expression in a function's `return` statement, with implicit
// conversions peeled off.
consteval m::info return_expr(m::info fn) {
  for (m::info s : m::statements_of(m::body_of(fn)))
    if (m::is_return_statement(s)) {
      m::info e = m::return_value_of(s);
      while (m::is_cast(e))
        e = m::operands_of(e).front();
      return e;
    }
  throw "no return statement";
}

constexpr m::info CMP = return_expr(^^f_cmp);
constexpr m::info OR = return_expr(^^f_or);
constexpr m::info AND = return_expr(^^f_and);
constexpr m::info NOT = return_expr(^^f_not);

// Comparison: fine on both compilers.
static_assert(m::is_binary_operator(CMP));

// Logical operators: the operator and operands can be queried on GCC too.
static_assert(m::expression_operator_of(OR) == m::operators::op_pipe_pipe);
static_assert(m::expression_operator_of(AND) ==
              m::operators::op_ampersand_ampersand);
static_assert(m::expression_operator_of(NOT) == m::operators::op_exclamation);
static_assert(m::operands_of(OR).size() == 2);
static_assert(m::operands_of(AND).size() == 2);
static_assert(m::operands_of(NOT).size() == 1);

int main() {}
