// constant_of of an id-expression gives the constant of the variable it names,
// on both compilers. An expression that names a constexpr variable in a
// reflected body can be folded through it, with no declaration_of in between.
//
// GCC used to throw "reflection represents an id-expression, not an entity"
// (#242). Fixed in CppReflectionAD/gcc#23. The fork has the same checks in
// g++.dg/reflect/id_expression3.C.
#include <ad/meta.h>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

struct P {
  double a;
};
inline constexpr double k = 2.5;
inline constexpr P kp{1.5};
inline const int ci = 7;

const double &f_k() { return k; }
const P &f_kp() { return kp; }
const int &f_ci() { return ci; }

static_assert(m::is_id_expression(return_expr(^^f_k)));
static_assert(m::constant_of(return_expr(^^f_k)) == m::constant_of(^^k),
              "a constexpr double");
static_assert([:m::constant_of(return_expr(^^f_k)):] == 2.5);

static_assert(m::is_id_expression(return_expr(^^f_kp)));
static_assert(m::constant_of(return_expr(^^f_kp)) == m::constant_of(^^kp),
              "a constexpr class object");
static_assert([:m::constant_of(return_expr(^^f_kp)):].a == 1.5);

static_assert(m::is_id_expression(return_expr(^^f_ci)));
static_assert(m::constant_of(return_expr(^^f_ci)) == m::constant_of(^^ci),
              "a const int with a constant initializer");
static_assert([:m::constant_of(return_expr(^^f_ci)):] == 7);

// The id-expression is still not the variable.
static_assert(return_expr(^^f_k) != ^^k);
static_assert(!m::is_variable(return_expr(^^f_k)));
static_assert(m::declaration_of(return_expr(^^f_k)) == ^^k);

int main() {}
