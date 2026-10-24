// TEST-PASSES-ON: clang
//
// Known GCC gap: a call to std::fabs is folded to an ABS_EXPR (#183).
//
// clang reflects `std::fabs(x)` as a call to std::fabs, which autograd.h
// registers as a primitive (OpKind::Abs). GCC folds the builtin into an
// ABS_EXPR that it reports as a unary operator, but expression_operator_of()
// then throws "code=abs_expr", so the engine can't lower it. Building with
// -fno-builtin avoids the fold. This keeps limit_algebra, is_continuous,
// absolute_normal_form and ad_control_flow_demo in tests/clang_only/.
//
// On GCC, tests/CMakeLists.txt expects the static_assert below to fail, with
// the error in fabs_reflection.fail.txt. The TEST-PASSES-ON line makes this an
// ordinary test on clang. When GCC fixes the gap, move this file to
// tests/common/, drop that line and delete the .fail.txt.
#include <ad/meta.h>

#include <cmath>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

double f_fabs(double x) { return std::fabs(x); }

static_assert(m::is_function_call(return_expr(^^f_fabs)),
              "std::fabs(x) is not a function call");

int main() {}
