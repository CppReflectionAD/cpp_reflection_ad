// TEST-PASSES-ON: clang
//
// Known GCC gap #220: a declaration with several declarators is split into one
// declaration statement per declarator.
//
// `const double a = x, b = a * a;` is one simple-declaration, so one
// declaration statement whose declarators_of has two elements. clang reflects
// it that way. GCC reflects it as two declaration statements of one variable
// each, so statements_of gives a different count, and different statements,
// for the same source on the two compilers.
//
// This is a problem: what reflection reports about a function body should
// depend on the source, not on the compiler. Reflection AD gets the same
// result either way, since it lowers each declarator in order, but code that
// counts or indexes statements, or maps them back to the source, does not.
//
// The asserts check both halves of the fix: one statement, and that statement
// listing every declarator. A GCC fix that merged the statements but still
// reported one variable per statement would fail the second.
//
// On GCC, tests/CMakeLists.txt expects all four multi-declarator static_asserts
// below to fail, with the errors in multiple_declarators_reflection.fail.txt.
// The TEST-PASSES-ON line makes this an ordinary test on clang, which passes
// them. When GCC fixes the gap, move this file to tests/common/, drop that line
// and delete the .fail.txt.
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
