// An `if` whose init-statement or condition declares a variable, or whose
// clause is a block that does, is the `if`, with the declarations as its first
// parts ([stmt.if]), on both compilers.
//
// GCC used to give the scope around such an `if` as a compound statement of
// the declarations and the `if` (#278). Fixed for `if` in
// CppReflectionAD/gcc#41, which has more checks in
// g++.dg/reflect/if_cond_scope1.C, and for `switch` in CppReflectionAD/gcc#47
// (switch_condition_scope.cpp).
//
// The engine still rejects any `if` in a body it differentiates.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int plain(int x) {
  if (x)
    sink(x);
  else
    sink(1);
  return 0;
}

inline int declares(int x) {
  if (int y = sink(x))
    sink(y);
  else
    sink(1);
  return 0;
}

inline int init_statement(int x) {
  if (int y = sink(x); int z = y)
    sink(z);
  else
    sink(1);
  return 0;
}

inline int block_declares(int x) {
  if (x) {
    int y = x;
    sink(y);
  } else
    sink(1);
  return 0;
}

template <typename T> int in_template(T x) {
  if (T y = x)
    sink(y);
  return 0;
}

// The `if` that is the first statement of fn's body.
consteval info first(info fn) { return m::statements_of(m::body_of(fn))[0]; }

consteval std::size_t parts(info s) { return m::statements_of(s).size(); }

consteval bool is_if_shape(info s, std::size_t n) {
  return m::is_statement(s) && !m::is_compound_statement(s) && parts(s) == n &&
         m::is_expression_statement(m::statements_of(s)[n - 1]);
}

static_assert(is_if_shape(first(^^plain), 3), "condition, then, else");
static_assert(is_if_shape(first(^^declares), 4),
              "declaration, condition, then, else");
static_assert(is_if_shape(first(^^init_statement), 5),
              "init-statement, declaration, condition, then, else");
static_assert(is_if_shape(first(^^block_declares), 3),
              "a block that declares: condition, then, else");
static_assert(
    m::is_compound_statement(m::statements_of(first(^^block_declares))[1]),
    "a block that declares: the block");
static_assert(is_if_shape(first(^^in_template<int>), 3),
              "in a template: declaration, condition, then");

int main() {
  return plain(0) + declares(0) + init_statement(0) + block_declares(0) +
         in_template(0);
}
