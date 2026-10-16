// Known GCC gap: `||`, `&&` and `!` are not classified as operators.
//
// For a comparison such as `x > 0.0`, both compilers report
// is_binary_operator == true. For the logical operators, clang reports
// is_binary_operator / is_unary_operator == true, but GCC reports false, even
// though its expression_operator_of() and operands_of() already handle these
// expressions (see tests/gcc_logical_operator_reflection.cpp).
//
// tests/CMakeLists.txt expects all three static_asserts below to fail, with the
// errors in logical_operator_reflection.fail.txt, so the test also fails if GCC
// fixes only some of them. clang checks the same asserts in
// tests/clang_only/logical_operator_reflection.cpp. When GCC fixes the gap,
// move that file to tests/ and delete this one and its .fail.txt.
#include <meta>

namespace m = std::meta;

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

static_assert(m::is_binary_operator(return_expr(^^f_or)),
              "|| is not a binary operator");
static_assert(m::is_binary_operator(return_expr(^^f_and)),
              "&& is not a binary operator");
static_assert(m::is_unary_operator(return_expr(^^f_not)),
              "! is not a unary operator");

int main() {}
