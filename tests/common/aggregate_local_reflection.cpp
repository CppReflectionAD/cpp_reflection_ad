// A local aggregate or array with a non-constant initializer is one
// declaration statement, with one declarator and one initializer, on both
// compilers: `T b = {x}; return b.v;` has two statements.
//
// GCC used to split such an initializer into the initializations of b's
// subobjects (#194): b's declaration had no initializer, or only the constant
// part of it, and was followed by a second statement, `b.v = x`, that
// is_declaration_statement accepted and declarators_of rejected. Fixed in
// CppReflectionAD/gcc#12. A constant initializer, a class with a
// constructor and a scalar were never split; they are controls. The fork has
// the same checks in g++.dg/reflect/aggregate_local_init1.C.
#include <ad/meta.h>

#include <string_view>

namespace m = ad::meta;

void destroyed(double) {}

struct T {
  double v;
};
struct P {
  double v, w;
};
struct N {
  T t;
  double w;
};
struct D {
  double v;
  ~D() { destroyed(v); }
};
struct Q {
  D d;
  double w;
};
struct C {
  C(double x) : v(x) {}
  double v;
};

double t_const(double) {
  T b = {1.0};
  return b.v;
}
double t_brace(double x) {
  T b = {x};
  return b.v;
}
double t_list(double x) {
  T b{x};
  return b.v;
}
double t_paren(double x) {
  T b(x);
  return b.v;
}
double t_temp(double x) {
  T b = T{x};
  return b.v;
}
double p_brace(double x) {
  P b = {x, x};
  return b.v;
}
double p_mixed(double x) {
  P b = {1.0, x};
  return b.w;
}
double n_brace(double x) {
  N b = {{x}, x};
  return b.w;
}
double d_brace(double x) {
  D b = {x};
  return b.v;
}
double q_brace(double x) {
  Q b = {D{x}, x};
  return b.w;
}
double arr_brace(double x) {
  double b[2] = {x, x};
  return b[0];
}
double c_brace(double x) {
  C b = {x};
  return b.v;
}
double scalar(double x) {
  double b{x};
  return b;
}
constexpr double cx(double x) {
  T b = {x};
  return b.v;
}
template <typename U> U tmpl(U x) {
  T b = {x};
  return b.v;
}
template double tmpl<double>(double);

// Whether fn's body is the declaration of b, with one initializer, and a
// return.
consteval bool one_initialized_declaration(m::info fn) {
  auto stmts = m::statements_of(m::body_of(fn));
  if (stmts.size() != 2 || !m::is_declaration_statement(stmts[0]) ||
      !m::is_return_statement(stmts[1]))
    return false;
  try {
    auto vars = m::declarators_of(stmts[0]);
    return vars.size() == 1 &&
           m::identifier_of(vars[0]) == std::string_view("b") &&
           m::initializers_of(stmts[0]).size() == 1;
  } catch (...) {
    return false;
  }
}

static_assert(one_initialized_declaration(^^t_const));
static_assert(one_initialized_declaration(^^t_brace),
              "`T b = {x};` is not one declaration with an initializer");
static_assert(one_initialized_declaration(^^t_list));
static_assert(one_initialized_declaration(^^t_paren));
static_assert(one_initialized_declaration(^^t_temp));
static_assert(one_initialized_declaration(^^p_brace));
static_assert(one_initialized_declaration(^^p_mixed));
static_assert(one_initialized_declaration(^^n_brace));
static_assert(one_initialized_declaration(^^d_brace));
static_assert(one_initialized_declaration(^^q_brace));
static_assert(one_initialized_declaration(^^arr_brace));
static_assert(one_initialized_declaration(^^c_brace));
static_assert(one_initialized_declaration(^^scalar));
static_assert(one_initialized_declaration(^^cx));
static_assert(one_initialized_declaration(^^tmpl<double>));

int main() {}
