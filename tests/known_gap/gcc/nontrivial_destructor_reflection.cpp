// TEST-PASSES-ON: clang
//
// Known GCC gap: in a function body that uses a class with a non-trivial
// destructor, GCC's statements_of() reports no return statement.
//
// The controls below give the same bodies a trivially destructible type, and
// both compilers find their returns. Give the type a destructor and GCC
// reports a different statement where the return was: after a local of that
// type (d_local_then_return) the return is wrapped in the local's cleanup and
// comes back as a <try_finally_expr> expression, and a body that is just
// `return make_d(x);` (d_return_call) comes back as an <init_expr> of the
// return slot. Clang reports both returns, and the returned call as a function
// call. This blocks reflecting code over heap-backed types such as a tensor
// with std::vector storage.
//
// GCC also reports `return make_t(x);` as a construct rather than a function
// call, even for the trivially destructible type, so once it finds the
// returns, the second gap assert may still fail, with a different error.
//
// What GCC reports for d_return_call also depends on the flags: with
// optimization and -g together (e.g. RelWithDebInfo's -O2 -g) it does find
// the return, as a construct like make_t's. tests/CMakeLists.txt appends -O0
// on GCC so the result, and the .fail.txt, don't depend on the build type.
//
// tests/CMakeLists.txt builds this file as an ordinary test on clang (see
// TEST-PASSES-ON above). On GCC it expects both gap asserts to fail, with the
// errors in nontrivial_destructor_reflection.fail.txt, so the test also fails
// if GCC fixes only one of them. When GCC fixes the gap, move this file to
// tests/common/, drop the TEST-PASSES-ON line and delete the .fail.txt.
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

// Controls: without a destructor, GCC finds both returns.
static_assert(has_return(^^t_local_then_return));
static_assert(has_return(^^t_return_call));

// The gap.
static_assert(has_return(^^d_local_then_return));
static_assert(m::is_function_call(return_expr(^^d_return_call)),
              "`return make_d(x);` is not a return of a function call");

int main() {}
