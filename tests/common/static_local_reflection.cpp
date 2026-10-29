// A static local with a dynamic initializer is one declaration statement,
// with its initializer, on both compilers: `static double b = x; return b;`
// has two statements.
//
// GCC used to declare b without an initializer and initialize it in a second
// statement, an `if` on b's guard variable that runs once (#230). Fixed in
// CppReflectionAD/gcc#19.
// A constant initializer was never split; it is a control. The fork has the
// same checks in g++.dg/reflect/static_local_init1.C.
#include <ad/meta.h>

#include <cstddef>
#include <string_view>

namespace m = ad::meta;

void destroyed(double) {}

struct C {
  constexpr C(double x) : v(x) {}
  double v;
};
struct D {
  D(double x) : v(x) {}
  ~D() { destroyed(v); }
  double v;
};
struct T {
  double v;
};

double s_const(double) {
  static C b(1.0);
  return b.v;
}
double s_scalar(double x) {
  static double b = x;
  return b;
}
double s_copy(double x) {
  static C b = x;
  return b.v;
}
double s_paren(double x) {
  static C b(x);
  return b.v;
}
double s_brace(double x) {
  static C b{x};
  return b.v;
}
double s_dtor(double x) {
  static D b(x);
  return b.v;
}
double s_aggr(double x) {
  static T b = {x};
  return b.v;
}
double s_tl(double x) {
  thread_local double b = x;
  return b;
}
template <typename U> U tmpl(U x) {
  static U b = x;
  return b;
}
template double tmpl<double>(double);

// Whether fn's body is the declaration of b, with an initializer, and a
// return.
consteval bool one_initialized_declaration(m::info fn) {
  auto stmts = m::statements_of(m::body_of(fn));
  if (stmts.size() != 2 || !m::is_declaration_statement(stmts[0]) ||
      !m::is_return_statement(stmts[1]))
    return false;
  try {
    auto vars = m::declarators_of(stmts[0]);
    auto inits = m::initializers_of(stmts[0]);
    return vars.size() == 1 &&
           m::identifier_of(vars[0]) == std::string_view("b") &&
           inits.size() == 1 && inits[0] != m::info{};
  } catch (...) {
    return false;
  }
}

// Whether the initializer of fn's first statement constructs b from n
// arguments, rather than calling the constructor.
consteval bool constructed_from(m::info fn, std::size_t n) {
  auto init = m::initializers_of(m::statements_of(m::body_of(fn))[0])[0];
  return !m::is_function_call(init) && m::operands_of(init).size() == n;
}

static_assert(one_initialized_declaration(^^s_const));
static_assert(one_initialized_declaration(^^s_scalar),
              "`static double b = x;` is not one declaration with an "
              "initializer");
static_assert(one_initialized_declaration(^^s_copy));
static_assert(one_initialized_declaration(^^s_paren));
static_assert(one_initialized_declaration(^^s_brace));
static_assert(one_initialized_declaration(^^s_dtor));
static_assert(one_initialized_declaration(^^s_aggr));
static_assert(one_initialized_declaration(^^s_tl));
static_assert(one_initialized_declaration(^^tmpl<double>));

static_assert(constructed_from(^^s_copy, 1));
static_assert(constructed_from(^^s_paren, 1));
static_assert(constructed_from(^^s_brace, 1));
static_assert(constructed_from(^^s_dtor, 1));

int main() {}
