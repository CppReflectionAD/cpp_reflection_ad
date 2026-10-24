// A local keeps its initializer when it follows a local with a non-trivial
// destructor, on both compilers: in `D a = make_d(x); D b = pass_d(a);`, b's
// initializer is the call.
//
// GCC used to lose the initializer of every local after one with a
// non-trivial destructor (#187). This, with nrvo_return_reflection.cpp, is
// what kept the tensor demo, whose ops take Tensor by value, clang-only.
#include <ad/meta.h>

#include <string_view>
#include <type_traits>

namespace m = ad::meta;

void destroyed(double) {}

struct T {
  double v;
};

struct D {
  double v;
  ~D() { destroyed(v); }
};

static_assert(std::is_trivially_destructible_v<T>);
static_assert(!std::is_trivially_destructible_v<D>);

T make_t(double x) { return {x}; }
T pass_t(T x) { return x; }
D make_d(double x) { return {x}; }
D pass_d(D x) { return x; }

double t_copy(double x) {
  T a = make_t(x);
  T b = pass_t(a);
  return b.v;
}

double d_copy(double x) {
  D a = make_d(x);
  D b = pass_d(a);
  return b.v;
}

// Whether every declaration statement in fn that declares `b` has an
// initializer.
consteval bool b_has_initializer(m::info fn) {
  bool found = false;
  for (m::info s : m::statements_of(m::body_of(fn))) {
    if (!m::is_declaration_statement(s) ||
        m::identifier_of(m::declarators_of(s)[0]) != std::string_view("b"))
      continue;
    found = true;
    try {
      m::initializers_of(s);
    } catch (...) {
      return false;
    }
  }
  return found;
}

static_assert(b_has_initializer(^^t_copy));
static_assert(b_has_initializer(^^d_copy),
              "b, initialized from a copy of a, has no initializer");

int main() {}
