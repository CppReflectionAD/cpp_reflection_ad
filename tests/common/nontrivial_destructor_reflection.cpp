// statements_of() must find the return statement in a function body that uses
// a class with a non-trivial destructor, and report a returned call as a call
// to the right function, as autograd.h finds primitives through callee_of().
//
// The GCC fork used to get this wrong: after a local of such a type
// (d_local_then_return) the return was wrapped in the local's cleanup and came
// back as a <try_finally_expr> expression, and `return make_d(x);`
// (d_return_call) came back as an <init_expr> of the return slot. Fixed in
// CppReflectionAD/gcc#2. The T functions are controls with a trivially
// destructible type. The fork has the same checks in
// g++.dg/reflect/statements_of_nontrivial_dtor1.C.
//
// GCC still gets nearby cases wrong: returning such a local by name, copying
// one into a by-value parameter, and reflecting any class-type local as two
// declaration statements, which the count comparison below can't see: #188,
// #187 and #186, with tests in tests/known_gap/gcc/.
#include <ad/meta.h>
#include <type_traits>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

void destroyed(double) {}

struct T {
  double v;
};

struct D {
  double v;
  ~D() { destroyed(v); }
};

// So the D cases can't quietly turn into copies of the T controls.
static_assert(std::is_trivially_destructible_v<T>);
static_assert(!std::is_trivially_destructible_v<D>);

T make_t(double x) { return {x}; }
D make_d(double x) { return {x}; }

double t_local_then_return(double x) {
  T a = make_t(x);
  return a.v;
}
T t_return_call(double x) { return make_t(x); }

double d_local_then_return(double x) {
  D a = make_d(x);
  return a.v;
}
D d_return_call(double x) { return make_d(x); }

// Throws, rather than returning false, if there is no return statement.
consteval bool has_return(m::info fn) {
  return m::is_return_statement(return_statement_of(fn));
}

static_assert(has_return(^^t_local_then_return));
static_assert(has_return(^^t_return_call));
static_assert(has_return(^^d_local_then_return));
static_assert(has_return(^^d_return_call));

// The local's destructor adds no statement.
static_assert(m::statements_of(m::body_of(^^d_local_then_return)).size() ==
              m::statements_of(m::body_of(^^t_local_then_return)).size());

static_assert(m::is_function_call(return_expr(^^t_return_call)),
              "`return make_t(x);` is not a return of a function call");
static_assert(m::is_function_call(return_expr(^^d_return_call)),
              "`return make_d(x);` is not a return of a function call");
static_assert(m::declaration_of(m::callee_of(return_expr(^^t_return_call))) ==
              ^^make_t);
static_assert(m::declaration_of(m::callee_of(return_expr(^^d_return_call))) ==
              ^^make_d);
static_assert(m::operands_of(return_expr(^^d_return_call)).size() == 1);

int main() {}
