// TEST-PASSES-ON: clang
//
// Known GCC gap: a local of class type initialized by a call is reflected as
// two declaration statements (#186).
//
// clang gives `T a = make_t(x); return a.v;` two statements: the declaration
// and the return. GCC adds a second declaration statement for `a`, holding the
// <init_expr> that initializes it, so every class-type local appears twice.
// The type needn't have a non-trivial destructor. The engine lowers both, so
// this alone doesn't break AD, but it builds each such local twice.
// tests/common/nontrivial_destructor_reflection.cpp compares these counts
// between class types, so it can't see this.
//
// On GCC, tests/CMakeLists.txt expects the static_assert below to fail, with
// the error in class_local_declaration_reflection.fail.txt. The TEST-PASSES-ON
// line makes this an ordinary test on clang. When GCC fixes the gap, move this
// file to tests/common/, drop that line and delete the .fail.txt.
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
