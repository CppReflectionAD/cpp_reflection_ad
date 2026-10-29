// TEST-PASSES-ON: clang
// Known GCC gap (#235): a declaration statement that declares no ordinary
// variable is missing from statements_of, or has the wrong kind. On clang each
// of these is a declaration statement that isn't a simple declaration, so
// declarators_of gives none. P4395R0 §3.3 excludes structured bindings from
// simple declarations because their names share one initializer; clang
// excludes them since CppReflectionAD/clang-p4395#8 (#203).
//
// On GCC, the binding is a statement of no kind, the using-declaration is an
// expression statement, and the others are missing. When GCC fixes the gap,
// the test fails because the file compiles: move it to tests/common/ and
// delete the .fail.txt.
#include <ad/meta.h>

namespace m = ad::meta;

namespace N {
inline double g(double x) { return x; }
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

int main() {}
