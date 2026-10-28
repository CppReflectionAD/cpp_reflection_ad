// A call that returns `const T&` reflects as a call, on both compilers.
//
// autograd.h registers std::max and std::min as primitives (OpKind::Max and
// OpKind::Min), by their callees. GCC used to report the dereference of the
// returned reference as a unary `*` applied to the call (#184).
#include <ad/meta.h>

#include <algorithm>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

double f_max(double x, double y) { return std::max(x, y); }
double f_min(double x, double y) { return std::min(x, y); }

static_assert(m::is_function_call(return_expr(^^f_max)),
              "std::max(x, y) is not a function call");
static_assert(m::is_function_call(return_expr(^^f_min)),
              "std::min(x, y) is not a function call");

int main() {}
