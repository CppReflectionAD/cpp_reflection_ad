// TEST-PASSES-ON: clang
//
// Known GCC gap: a local's initializer is lost when it passes another local
// with a non-trivial destructor to a function by value.
//
// In `D b = pass_d(a);`, the call copies `a` into a temporary that has to be
// destroyed. GCC then has no initializer for `b`: initializer_of() throws
// "reflection does not represent a variable declaration with initializer", so
// the engine can't lower the declaration. With a trivially destructible type
// (t_copy) the initializer is found. This is one of the two gaps that keep
// the tensor demo in tests/clang_only/: its ops take Tensor by value.
// nrvo_return_reflection.cpp is the other.
//
// On GCC, tests/CMakeLists.txt expects the static_assert on d_copy below to
// fail, with the error in copied_local_initializer_reflection.fail.txt. The
// TEST-PASSES-ON line makes this an ordinary test on clang. When GCC fixes the
// gap, move this file to tests/common/, drop that line and delete the
// .fail.txt.
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
