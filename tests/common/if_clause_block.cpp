// A block that is the `then` or `else` of an `if` is a compound statement
// ([stmt.block]), on both compilers, whatever it holds.
//
// GCC used to drop such a block when it held one statement, so statements_of
// on the `if` gave that statement instead (#269). A block of two statements,
// or none, was kept. Fixed in CppReflectionAD/gcc#39, which has these checks,
// and more, in g++.dg/reflect/if_clause_block1.C.
//
// The engine still rejects any `if` in a body it differentiates.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int one(int x) {
  if (x) {
    sink(x);
  }
  return 0;
}

inline int two(int x) {
  if (x) {
    sink(x);
    sink(1);
  }
  return 0;
}

inline int empty(int x) {
  if (x) {
  }
  return 0;
}

inline int else_one(int x) {
  if (x)
    sink(1);
  else {
    sink(x);
  }
  return 0;
}

inline int no_block(int x) {
  if (x)
    sink(x);
  else
    sink(1);
  return 0;
}

inline int constant(int x) {
  if (true) {
    sink(x);
  }
  return 0;
}

template <typename T> int in_template(T x) {
  if (x) {
    sink(x);
  }
  return 0;
}

// Part i of the `if` that is the first statement of fn's body: its condition,
// then its `then`, then its `else`.
consteval info part(info fn, std::size_t i) {
  return m::statements_of(m::statements_of(m::body_of(fn))[0])[i];
}

consteval std::size_t size(info s) { return m::statements_of(s).size(); }

static_assert(m::is_compound_statement(part(^^one, 1)), "one statement");
static_assert(size(part(^^one, 1)) == 1, "one statement");
static_assert(m::is_expression_statement(m::statements_of(part(^^one, 1))[0]),
              "one statement");
static_assert(m::is_compound_statement(part(^^two, 1)), "two statements");
static_assert(size(part(^^two, 1)) == 2, "two statements");
static_assert(m::is_compound_statement(part(^^empty, 1)), "an empty block");
static_assert(size(part(^^empty, 1)) == 0, "an empty block");

static_assert(m::is_expression_statement(part(^^else_one, 1)),
              "the `then` of an `else` block");
static_assert(m::is_compound_statement(part(^^else_one, 2)), "an `else` block");
static_assert(size(part(^^else_one, 2)) == 1, "an `else` block");
static_assert(m::is_expression_statement(part(^^no_block, 1)), "no block");
static_assert(m::is_expression_statement(part(^^no_block, 2)), "no block");

static_assert(m::is_compound_statement(part(^^constant, 1)),
              "a constant condition");
static_assert(m::is_compound_statement(part(^^in_template<int>, 1)),
              "in a template");

int main() {
  return one(0) + two(0) + empty(0) + else_one(1) + no_block(1) + constant(0) +
         in_template(0);
}
