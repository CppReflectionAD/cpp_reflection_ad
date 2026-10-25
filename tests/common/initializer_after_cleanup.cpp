// A second local of a class with a non-trivial destructor keeps its
// initializer, on both compilers: in `D a = make_d(x); D b = make_d(a.v);`,
// b's initializer is the call.
//
// GCC used to lose it (#187): the first local's destructor makes the rest of
// the block the body of a TRY_FINALLY_EXPR, which the fork's search for a
// local's initialization didn't enter. copied_local_initializer_reflection.cpp
// covers a copy of the first local (`pass_d(a)`); this covers a member of it.
// It is what lets a reflected body name more than one intermediate of a
// heap-backed type, as nn_mlp.t.cpp's heap_loss does.
#include <ad/meta.h>

#include <type_traits>

namespace m = ad::meta;

struct T {
  double v;
};

struct D {
  double v;
  ~D() {}
};

static_assert(std::is_trivially_destructible_v<T>);
static_assert(!std::is_trivially_destructible_v<D>);

T make_t(double x) { return {x}; }
D make_d(double x) { return {x}; }

double two_t(double x) {
  T a = make_t(x);
  T b = make_t(a.v);
  return b.v;
}

double two_d(double x) {
  D a = make_d(x);
  D b = make_d(a.v);
  return b.v;
}

// Whether each declarator in fn's body has an initializer. Throws, rather
// than returning false, if one has none.
consteval bool every_local_has_initializer(m::info fn) {
  for (m::info s : m::statements_of(m::body_of(fn)))
    if (m::is_declaration_statement(s) &&
        m::initializers_of(s).size() != m::declarators_of(s).size())
      return false;
  return true;
}

// Without a destructor.
static_assert(every_local_has_initializer(^^two_t));

// With one.
static_assert(every_local_has_initializer(^^two_d));

int main() {}
