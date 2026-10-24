// TEST-PASSES-ON: clang
//
// Known GCC gap: `return a;` of a local with a non-trivial destructor loses
// the local's declaration (#188).
//
// When a function returns a local of such a type by name (the named return
// value optimization), clang still gives two statements: `a`'s declaration and
// the return. GCC no longer reports the declaration as a declaration
// statement, and adds <nop_expr>, <try_catch_expr> and <call_expr> statements
// for the local's cleanup, so the engine rejects the body as "unsupported
// statement". Returning `a.v` or a call is fine (see
// tests/common/nontrivial_destructor_reflection.cpp). This is one of the two
// gaps that keep the tensor demo, whose loss() ends in `return L;`, in
// tests/clang_only/; copied_local_initializer_reflection.cpp is the other.
//
// On GCC, tests/CMakeLists.txt expects both static_asserts below to fail, with
// the errors in nrvo_return_reflection.fail.txt. The TEST-PASSES-ON line makes
// this an ordinary test on clang. When GCC fixes the gap, move this file to
// tests/common/, drop that line and delete the .fail.txt.
#include <ad/meta.h>

#include <type_traits>

namespace m = ad::meta;

void destroyed(double) {}

struct D {
  double v;
  ~D() { destroyed(v); }
};

static_assert(!std::is_trivially_destructible_v<D>);

D make_d(double x) { return {x}; }

D return_local(double x) {
  D a = make_d(x);
  return a;
}

static_assert(
    m::is_declaration_statement(
        m::statements_of(m::body_of(^^return_local))[0]),
    "the returned local's declaration is not a declaration statement");
static_assert(m::statements_of(m::body_of(^^return_local)).size() == 2,
              "returning a local adds statements for its cleanup");

int main() {}
