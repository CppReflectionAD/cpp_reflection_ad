// statements_of() must find the return statement in a function body that uses
// a class with a non-trivial destructor, and report a returned call as a
// function call.
//
// The GCC fork used to get this wrong: after a local of such a type
// (d_local_then_return) the return was wrapped in the local's cleanup and came
// back as a <try_finally_expr> expression, and `return make_d(x);`
// (d_return_call) came back as an <init_expr> of the return slot. Fixed in
// CppReflectionAD/gcc#2. The T functions are controls with a trivially
// destructible type.
#include <meta>

#include "reflection/return_expr.hpp"

namespace m = std::meta;

struct T {
  double v;
};

struct D {
  double v;
  ~D() {}
};

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

static_assert(m::is_function_call(return_expr(^^t_return_call)),
              "`return make_t(x);` is not a return of a function call");
static_assert(m::is_function_call(return_expr(^^d_return_call)),
              "`return make_d(x);` is not a return of a function call");

int main() {}
