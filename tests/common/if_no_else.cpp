// An `if` with no `else` has two parts, its condition and its `then`
// ([stmt.if]), on both compilers.
//
// GCC used to give it a third part, an empty statement, which
// false_expression_of also gave as the `else` (#270). Fixed in
// CppReflectionAD/gcc#38, which has these checks, and more, in
// g++.dg/reflect/if_no_else1.C. The checks of false_expression_of stay there:
// the clang fork's takes only an expression, not an `if`.
//
// The engine still rejects any `if` in a body it differentiates.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int no_else(int x) {
  if (x)
    sink(x);
  return 0;
}

inline int with_else(int x) {
  if (x)
    sink(x);
  else
    sink(2);
  return 0;
}

// An `else` that is a null statement, which GCC's -Wempty-body warns about.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wempty-body"
inline int null_else(int x) {
  if (x)
    sink(x);
  else
    ;
  return 0;
}
#pragma GCC diagnostic pop

inline int block_no_else(int x) {
  if (x) {
    sink(x);
    sink(1);
  }
  return 0;
}

inline int else_if_no_else(int x) {
  if (x)
    sink(x);
  else if (x + 1)
    sink(1);
  return 0;
}

template <typename T> int in_template(T x) {
  if (x)
    sink(x);
  return 0;
}

// The `if` that is the first statement of fn's body.
consteval info first_if(info fn) { return m::statements_of(m::body_of(fn))[0]; }

consteval std::size_t parts(info s) { return m::statements_of(s).size(); }

static_assert(parts(first_if(^^no_else)) == 2, "no `else`");
static_assert(
    m::is_expression_statement(m::statements_of(first_if(^^no_else))[1]),
    "no `else`: the `then` is the last part");
static_assert(parts(first_if(^^with_else)) == 3, "an `else`");
static_assert(parts(first_if(^^null_else)) == 3, "an `else` that is `;`");
static_assert(parts(first_if(^^block_no_else)) == 2, "a block and no `else`");
static_assert(parts(first_if(^^in_template<int>)) == 2, "in a template");

// An `else if` is the `else` of the outer `if`, and the inner one has none.
static_assert(parts(first_if(^^else_if_no_else)) == 3, "an `else if`");
static_assert(parts(m::statements_of(first_if(^^else_if_no_else))[2]) == 2,
              "the inner `if` of an `else if`");

int main() {
  return no_else(0) + with_else(0) + null_else(0) + block_no_else(0) +
         else_if_no_else(0) + in_template(0);
}
