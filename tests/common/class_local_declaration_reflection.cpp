// A local of class type initialized by a call is one declaration statement,
// on both compilers: `T a = make_t(x); return a.v;` has two statements.
//
// GCC used to add a second declaration statement for `a`, holding the
// <init_expr> that initializes it, so the engine built every class-type local
// twice (#186). tests/common/nontrivial_destructor_reflection.cpp compares
// these counts between class types, so it couldn't see this.
#include <ad/meta.h>

namespace m = ad::meta;

struct T {
  double v;
};

T make_t(double x) { return {x}; }

double t_local(double x) {
  T a = make_t(x);
  return a.v;
}

static_assert(m::statements_of(m::body_of(^^t_local)).size() == 2,
              "a class-type local is more than one declaration statement");

int main() {}
