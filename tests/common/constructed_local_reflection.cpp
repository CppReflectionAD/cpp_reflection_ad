// A local of class type that a constructor initializes is one declaration
// statement, whose initializer constructs it, on both compilers:
// `C b(x); return b.v;` has two statements.
//
// GCC used to declare b without an initializer and call the constructor in a
// second statement, `C::C(&b, x)` (#215). For a named return value returned
// by invisible reference, only that call was left. Fixed in
// CppReflectionAD/gcc#16.
// Copy-initialization and a constant initialization were never split; they
// are controls. The fork has the same checks in
// g++.dg/reflect/ctor_local_init1.C.
#include <ad/meta.h>

#include <cstddef>
#include <string_view>

namespace m = ad::meta;

void destroyed(double) {}

struct C {
  constexpr C(double x) : v(x) {}
  double v;
};
struct C0 {
  C0() : v(2.0) {}
  double v;
};
struct D {
  D(double x) : v(x) {}
  ~D() { destroyed(v); }
  double v;
};
struct E {
  constexpr E(double x, double y) : v(x), w(y) {}
  double v, w;
};

double c_copy(double x) {
  C b = x;
  return b.v;
}
double c_const(double) {
  C b(1.0);
  return b.v;
}
double c_paren(double x) {
  C b(x);
  return b.v;
}
double c_brace(double x) {
  C b{x};
  return b.v;
}
double c0_default(double x) {
  C0 b;
  return b.v * x;
}
double c0_brace(double x) {
  C0 b{};
  return b.v * x;
}
double d_paren(double x) {
  D b(x);
  return b.v;
}
double e_paren(double x, double y) {
  E b(x, y);
  return b.v;
}
double e_brace(double x, double y) {
  E b{x, y};
  return b.w;
}
C nrv_paren(double x) {
  C b(x);
  return b;
}
D nrv_d_paren(double x) {
  D b(x);
  return b;
}
D nrv_d_copy(double x) {
  D b = x;
  return b;
}
constexpr double cx(double x) {
  C b(x);
  return b.v;
}
template <typename U> U tmpl(U x) {
  C b(x);
  return b.v;
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

static_assert(one_initialized_declaration(^^c_copy));
static_assert(one_initialized_declaration(^^c_const));
static_assert(one_initialized_declaration(^^c_paren),
              "`C b(x);` is not one declaration with an initializer");
static_assert(one_initialized_declaration(^^c_brace));
static_assert(one_initialized_declaration(^^c0_default));
static_assert(one_initialized_declaration(^^c0_brace));
static_assert(one_initialized_declaration(^^d_paren));
static_assert(one_initialized_declaration(^^e_paren));
static_assert(one_initialized_declaration(^^e_brace));
static_assert(one_initialized_declaration(^^nrv_paren));
static_assert(one_initialized_declaration(^^nrv_d_paren));
static_assert(one_initialized_declaration(^^nrv_d_copy));
static_assert(one_initialized_declaration(^^cx));
static_assert(one_initialized_declaration(^^tmpl<double>));

static_assert(constructed_from(^^c_copy, 1));
static_assert(constructed_from(^^c_paren, 1));
static_assert(constructed_from(^^c_brace, 1));
static_assert(constructed_from(^^c0_default, 0));
static_assert(constructed_from(^^c0_brace, 0));
static_assert(constructed_from(^^d_paren, 1));
static_assert(constructed_from(^^e_paren, 2));
static_assert(constructed_from(^^e_brace, 2));
static_assert(constructed_from(^^nrv_d_paren, 1));

int main() {}
