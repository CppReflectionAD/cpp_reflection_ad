// TEST-PASSES-ON: clang
//
// Known GCC gap: in a function body, a second local of a class with a
// non-trivial destructor has no initializer. initializer_of() throws "does not
// represent a variable declaration with initializer" for it; the first such
// local, and locals of a trivially destructible type, are fine. Clang gives
// every initializer.
//
// The first local's destructor makes the rest of the block the body of a
// TRY_FINALLY_EXPR. statements_of() looks inside it, but the walker in the
// fork's reflect.cc that finds a local's initialization
// (find_var_initializer_in_scope) only enters STATEMENT_LISTs and BIND_EXPRs,
// so it never reaches the second local's.
//
// This blocks reflecting code over heap-backed types (e.g. a tensor with
// std::vector storage) that names more than one intermediate result, such as
// `HeapVec h = ...; HeapVec y = ...;`.
//
// tests/CMakeLists.txt builds this file as an ordinary test on clang. On GCC
// it expects the assert on two_d to fail, with the errors in
// initializer_after_cleanup.fail.txt. When GCC fixes the gap, move this file
// to tests/common/, drop the TEST-PASSES-ON line and delete the .fail.txt.
#include <ad/meta.h>

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

// Control: without a destructor, GCC finds both initializers.
static_assert(every_local_has_initializer(^^two_t));

// The gap.
static_assert(every_local_has_initializer(^^two_d));

int main() {}
