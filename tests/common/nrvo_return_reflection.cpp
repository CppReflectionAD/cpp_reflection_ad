// Returning a local with a non-trivial destructor by name (the named return
// value optimization) gives two statements, on both compilers: the local's
// declaration and the return.
//
// GCC used to drop the declaration and add <nop_expr>, <try_catch_expr> and
// <call_expr> statements for the local's cleanup, so the engine rejected the
// body as "unsupported statement" (#188). This, with
// copied_local_initializer_reflection.cpp, is what kept the tensor demo,
// whose loss() ends in `return L;`, clang-only.
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
