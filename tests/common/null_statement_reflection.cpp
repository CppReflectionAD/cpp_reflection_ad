// A null statement is an expression statement with no expression ([stmt.expr]),
// so a reflected body has one statement for each `;`, on both compilers.
//
// GCC used to leave it out of statements_of, and clang gave it no statement
// kind (#237). Fixed in CppReflectionAD/gcc#24 and
// CppReflectionAD/clang-p4395#9, which have the same checks in
// g++.dg/reflect/null_statement1.C and statement-reflection.pass.cpp.
#include <ad/meta.h>

namespace m = ad::meta;

consteval std::meta::info stmt(std::meta::info fn, std::size_t i) {
  return m::statements_of(m::body_of(fn))[i];
}

inline double f_null(double x) {
  ;
  return x;
}
inline double f_two(double x) {
  ;
  ;
  return x;
}
inline double f_end(double x) {
  return x;
  ;
}
template <typename T> T t_null(T x) {
  ;
  return x;
}

static_assert(m::statements_of(m::body_of(^^f_null)).size() == 2);
static_assert(m::is_expression_statement(stmt(^^f_null, 0)));
static_assert(!m::is_declaration_statement(stmt(^^f_null, 0)));
static_assert(m::is_return_statement(stmt(^^f_null, 1)));

static_assert(m::statements_of(m::body_of(^^f_two)).size() == 3);
static_assert(m::is_expression_statement(stmt(^^f_two, 0)));
static_assert(m::is_expression_statement(stmt(^^f_two, 1)));
static_assert(m::is_return_statement(stmt(^^f_two, 2)));

static_assert(m::statements_of(m::body_of(^^f_end)).size() == 2);
static_assert(m::is_return_statement(stmt(^^f_end, 0)));
static_assert(m::is_expression_statement(stmt(^^f_end, 1)));

static_assert(m::statements_of(m::body_of(^^t_null<double>)).size() == 2);
static_assert(m::is_expression_statement(stmt(^^t_null<double>, 0)));
static_assert(m::is_return_statement(stmt(^^t_null<double>, 1)));

int main() { return 0; }
