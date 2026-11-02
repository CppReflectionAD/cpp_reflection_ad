// A declaration with several declarators is one declaration statement, on
// both compilers.
//
// `const double a = x, b = a * a;` is one simple-declaration, so one
// declaration statement whose declarators_of has two elements. GCC used to
// reflect it as two declaration statements of one variable each (#220), so
// statements_of gave a different count, and different statements, for the
// same source on the two compilers. Fixed in CppReflectionAD/gcc#17.
//
// The asserts check both halves of the fix: one statement, and that statement
// listing every declarator. The fork has the same checks in
// g++.dg/reflect/multi_declarator1.C.
#include <ad/meta.h>

namespace m = ad::meta;

constexpr double two_declarators(double x) {
  const double a = x, b = a * a;
  return a + b;
}

constexpr double three_declarators(double x, double y) {
  const double a = 2, b = 3, c = x * y;
  return a * x + b * y + c;
}

// Control: separate declarations are separate statements on both compilers.
constexpr double separate_declarations(double x) {
  const double a = x;
  const double b = a * a;
  return a + b;
}

static_assert(m::statements_of(m::body_of(^^separate_declarations)).size() ==
              3);

static_assert(m::statements_of(m::body_of(^^two_declarators)).size() == 2,
              "`a = x, b = a * a` is not one declaration statement");
static_assert(m::statements_of(m::body_of(^^three_declarators)).size() == 2,
              "`a = 2, b = 3, c = x * y` is not one declaration statement");

static_assert(
    m::declarators_of(m::statements_of(m::body_of(^^two_declarators))[0])
            .size() == 2,
    "`a = x, b = a * a` does not list both declarators");
static_assert(
    m::declarators_of(m::statements_of(m::body_of(^^three_declarators))[0])
            .size() == 3,
    "`a = 2, b = 3, c = x * y` does not list all three declarators");

int main() {}
