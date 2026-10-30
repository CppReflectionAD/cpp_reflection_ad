// ad_block_declarations.cpp — a reflected body with a block declaration that
// declares no variable: an alias, a typedef, a static_assert, a
// using-declaration or -directive, a local class or enumeration, a namespace
// alias, a using-enum-declaration or a consteval block.
//
// Each is a declaration statement with no declarators on both compilers
// (#235), so the engine passes over it and differentiates the rest.

#include <test_simple_include.hpp>

#include <ad/autograd.h>
#include <ad/forward_derivative.h>
#include <ad/reverse_derivative.h>

namespace N {
constexpr double sq(double x) { return x * x; }
enum class E { a, b };
} // namespace N

constexpr double f_alias(double x) {
  using D = double;
  const D y = x * x;
  return y;
}
constexpr double f_typedef(double x) {
  typedef double D;
  const D y = x * x;
  return y;
}
constexpr double f_assert(double x) {
  static_assert(sizeof(double) == 8);
  return x * x;
}
constexpr double f_using(double x) {
  using N::sq;
  return sq(x);
}
constexpr double f_directive(double x) {
  using namespace N;
  return sq(x);
}
constexpr double f_class(double x) {
  struct L {
    double v;
  };
  return x * x;
}
constexpr double f_enum(double x) {
  enum E { a, b };
  return x * x;
}
constexpr double f_namespace_alias(double x) {
  namespace M = N;
  return M::sq(x);
}
constexpr double f_using_enum(double x) {
  using enum N::E;
  return x * x;
}
constexpr double f_consteval_block(double x) {
  consteval {}
  return x * x;
}
// Several, between declarations of variables.
constexpr double f_mixed(double x, double y) {
  using D = double;
  const D a = x * y;
  static_assert(true);
  using N::sq;
  const double b = sq(a);
  return a + b; // xy + x^2 y^2
}

int main() {
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_alias, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_typedef, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_assert, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_using, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_directive, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_class, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_enum, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_namespace_alias, 0>(3.0)), 6.0,
                  1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_using_enum, 0>(3.0)), 6.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^f_consteval_block, 0>(3.0)), 6.0,
                  1e-12);

  auto g = ad::gradient_reverse<^^f_mixed>(3.0, 5.0);
  EXPECT_NEAR_ABS(g[0], 5.0 + 2 * 3.0 * 25.0, 1e-12); // y + 2xy^2
  EXPECT_NEAR_ABS(g[1], 3.0 + 2 * 9.0 * 5.0, 1e-12);  // x + 2x^2 y

  TEST_END;
}
