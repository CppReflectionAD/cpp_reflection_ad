// A block declaration that declares no ordinary variable is one declaration
// statement that isn't a simple declaration, on both compilers, so
// declarators_of gives none. P4395R0 §3.3 excludes structured bindings from
// simple declarations because their names share one initializer; clang
// excludes them since CppReflectionAD/clang-p4395#8 (#203).
//
// GCC used to reflect the binding as a statement of no kind and the
// using-declaration as an expression statement, and to leave the others out
// of statements_of (#235). Fixed in CppReflectionAD/gcc#20. The fork has
// the same checks, and template instances of each, in
// g++.dg/reflect/block_decl[123].C.
#include <ad/meta.h>

namespace m = ad::meta;

namespace N {
inline double g(double x) { return x; }
enum class E { a, b };
} // namespace N
struct P {
  double a, b;
};

double f_binding(P p) {
  auto [a, b] = p;
  return a + b;
}
double f_alias(double x) {
  using D = double;
  return D(x);
}
double f_typedef(double x) {
  typedef double D;
  return D(x);
}
double f_assert(double x) {
  static_assert(true);
  return x;
}
double f_using(double x) {
  using N::g;
  return g(x);
}
double f_directive(double x) {
  using namespace N;
  return g(x);
}
double f_class(double x) {
  struct L {
    double v;
  };
  return L{x}.v;
}
double f_enum(double x) {
  enum E { a, b };
  return x + int(b);
}
double f_namespace_alias(double x) {
  namespace M = N;
  return M::g(x);
}
double f_using_enum(double x) {
  using enum N::E;
  return x + int(b);
}
double f_consteval_block(double x) {
  consteval {}
  return x;
}

// Whether fn's body is a declaration statement with no declarators, then a
// return.
consteval bool declaration_then_return(m::info fn) {
  auto stmts = m::statements_of(m::body_of(fn));
  return stmts.size() == 2 && m::is_declaration_statement(stmts[0]) &&
         !m::is_expression_statement(stmts[0]) &&
         !m::is_simple_declaration(stmts[0]) &&
         m::declarators_of(stmts[0]).empty() &&
         m::is_return_statement(stmts[1]);
}

static_assert(declaration_then_return(^^f_binding),
              "a structured binding is not one non-simple declaration");
static_assert(declaration_then_return(^^f_alias),
              "an alias declaration is not one non-simple declaration");
static_assert(declaration_then_return(^^f_typedef),
              "a typedef is not one non-simple declaration");
static_assert(declaration_then_return(^^f_assert),
              "a static_assert is not one non-simple declaration");
static_assert(declaration_then_return(^^f_using),
              "a using-declaration is not one non-simple declaration");
static_assert(declaration_then_return(^^f_directive),
              "a using-directive is not one non-simple declaration");
static_assert(declaration_then_return(^^f_class),
              "a local class is not one non-simple declaration");
static_assert(declaration_then_return(^^f_enum),
              "a local enumeration is not one non-simple declaration");
static_assert(declaration_then_return(^^f_namespace_alias),
              "a namespace alias is not one non-simple declaration");
static_assert(declaration_then_return(^^f_using_enum),
              "a using-enum-declaration is not one non-simple declaration");
static_assert(declaration_then_return(^^f_consteval_block),
              "a consteval block is not one non-simple declaration");

int main() {}
