// ad::meta::operands_of on a call gives the arguments only, on both compilers.
//
// GCC's own operands_of lists the callee first when the call has one, and
// ad/meta.h drops it. A call to an internal function (__builtin_launder) has
// none, so there the first operand is an argument and must stay.

#include <ad/meta.h>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

double g(double x) { return x; }
double h() { return 1.0; }
double (*fp)(double) = g;

double f_direct(double x) { return g(x); }
double f_nullary() { return h(); }
double f_pointer(double x) { return fp(x); }
int *f_launder(int *p) { return __builtin_launder(p); }

static_assert(m::operands_of(return_expr(^^f_direct)).size() == 1);
static_assert(m::operands_of(return_expr(^^f_nullary)).empty());
static_assert(m::operands_of(return_expr(^^f_pointer)).size() == 1);
static_assert(m::operands_of(return_expr(^^f_launder)).size() == 1);

int main() {}
