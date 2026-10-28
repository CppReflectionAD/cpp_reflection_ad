// A call to std::fabs reflects as a call, on both compilers.
//
// autograd.h registers std::fabs as a primitive (OpKind::Abs), by its callee.
// GCC used to fold the builtin into an ABS_EXPR, which it reported as a unary
// operator that expression_operator_of() rejects (#183).
#include <ad/meta.h>

#include <cmath>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

double f_fabs(double x) { return std::fabs(x); }

static_assert(m::is_function_call(return_expr(^^f_fabs)),
              "std::fabs(x) is not a function call");

int main() {}
