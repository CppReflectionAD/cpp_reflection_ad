// TEST-PASSES-ON: clang
//
// Known GCC gap: a call that returns `const T&` is reflected as a unary `*`
// applied to the call (#184).
//
// clang reflects `std::max(x, y)` and `std::min(x, y)` as calls, which
// autograd.h registers as primitives (OpKind::Max and OpKind::Min). GCC wraps
// the call in the dereference of the reference it returns, so the engine sees
// a unary operator it doesn't support. This keeps ad_control_flow_demo in
// tests/clang_only/, and std::max and std::min fail on GCC in any reflected
// function.
//
// On GCC, tests/CMakeLists.txt expects both static_asserts below to fail, with
// the errors in const_ref_return_reflection.fail.txt. The TEST-PASSES-ON line
// makes this an ordinary test on clang. When GCC fixes the gap, move this file
// to tests/common/, drop that line and delete the .fail.txt.
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
