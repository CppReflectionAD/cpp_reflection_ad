// Minimal reproducer for a GCC reflection bug: UBSan instrumentation leaks into
// the expression trees seen by std::meta.
//
// Build with: g++ -std=c++2c -freflection -fsanitize=float-divide-by-zero
//
// Without the sanitizer this compiles. With it, the right operand of `1.0 / x`
// is no longer reported as a variable reference, so the static_assert fails.
// The UBSan checks shift, integer-divide-by-zero, signed-integer-overflow,
// null, alignment and vptr break reflection on other expressions in the same
// way. They are disabled for GCC in the top-level CMakeLists.txt and only this
// file is compiled with one of them, in Debug.
#include <meta>

inline double f(double x) { return 1.0 / x; }

consteval bool rhs_is_variable_reference() {
  auto ret = std::meta::statements_of(std::meta::body_of(^^f))[0];
  auto ops = std::meta::operands_of(std::meta::return_value_of(ret));
  return std::meta::is_variable_reference(ops[1]);
}

static_assert(rhs_is_variable_reference());

int main() {}
