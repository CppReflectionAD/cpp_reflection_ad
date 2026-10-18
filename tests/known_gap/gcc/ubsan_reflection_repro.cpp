// Known GCC gap: UBSan instrumentation leaks into the expression trees seen by
// std::meta. From #92's minimal reproducer.
//
// Without -fsanitize=float-divide-by-zero this compiles. With it, the right
// operand of `1.0 / x` is no longer reported as a variable reference, so the
// static_assert fails. The UBSan checks shift, integer-divide-by-zero,
// signed-integer-overflow, null, alignment and vptr break reflection on other
// expressions in the same way.
//
// tests/CMakeLists.txt builds this file with the sanitizer flag and expects the
// error in ubsan_reflection_repro.fail.txt. When GCC fixes the gap, the test
// fails because the file compiles: move it to tests/gcc_only/ (without the
// flag) and delete the .fail.txt. Then drop the -fno-sanitize= list that
// REFLECTION_AD_SANITIZE adds for GCC in cmake/reflect_flags.cmake.
#include <meta>

inline double f(double x) { return 1.0 / x; }

consteval bool rhs_is_variable_reference() {
  auto ret = std::meta::statements_of(std::meta::body_of(^^f))[0];
  auto ops = std::meta::operands_of(std::meta::return_value_of(ret));
  return std::meta::is_variable_reference(ops[1]);
}

static_assert(rhs_is_variable_reference(),
              "UBSan instrumentation hides the variable reference in 1.0 / x");

int main() {}
