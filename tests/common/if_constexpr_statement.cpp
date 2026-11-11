// An `if constexpr` is one statement in the block it is in, on both
// compilers, whichever of its statements is discarded and whatever else is in
// the block (P4395R0 §3.2).
//
// GCC used to replace it by the statement its condition selects, so
// statements_of gave that statement, or that statement's statements, or
// nothing (#250). Fixed in CppReflectionAD/gcc#32, which has these checks,
// and more, in g++.dg/reflect/if_constexpr1.C.
//
// The engine still rejects an `if constexpr` in a body it differentiates, as
// it does any `if`, now on both compilers.
#include <ad/meta.h>

#include <cstddef>
#include <type_traits>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

template <typename T> void whole_body(T) {
  if constexpr (std::is_arithmetic_v<T>) {
    sink(1);
    sink(2);
  } else
    sink(3);
}

template <typename T> void not_taken(T) {
  if constexpr (!std::is_arithmetic_v<T>)
    sink(1);
}

template <typename T> int then_return(T) {
  if constexpr (std::is_arithmetic_v<T>) {
    sink(1);
    sink(2);
  }
  return 0;
}

template <typename T> int one_then_return(T) {
  if constexpr (std::is_arithmetic_v<T>) {
    sink(1);
  }
  return 0;
}

inline int not_a_template(int x) {
  if constexpr (sizeof(int) >= 2)
    sink(x);
  return x;
}

consteval std::size_t count(info fn) {
  return m::statements_of(m::body_of(fn)).size();
}

consteval info first(info fn) { return m::statements_of(m::body_of(fn))[0]; }

// A statement of no kind the paper names: not an expression statement, a
// block, a return or a declaration.
consteval bool other_statement(info s) {
  return m::is_statement(s) && !m::is_expression(s) &&
         !m::is_expression_statement(s) && !m::is_compound_statement(s) &&
         !m::is_return_statement(s) && !m::is_declaration_statement(s);
}

static_assert(count(^^whole_body<int>) == 1, "the whole body");
static_assert(other_statement(first(^^whole_body<int>)), "the whole body");
static_assert(count(^^not_taken<int>) == 1, "no statement taken");
static_assert(other_statement(first(^^not_taken<int>)), "no statement taken");
static_assert(count(^^then_return<int>) == 2,
              "a two-statement block, then a return");
static_assert(other_statement(first(^^then_return<int>)),
              "a two-statement block, then a return");
static_assert(count(^^one_then_return<int>) == 2,
              "a one-statement block, then a return");
static_assert(other_statement(first(^^one_then_return<int>)),
              "a one-statement block, then a return");
static_assert(count(^^not_a_template) == 2, "outside a template");
static_assert(other_statement(first(^^not_a_template)), "outside a template");

int main() {
  whole_body(1);
  not_taken(1);
  return then_return(1) + one_then_return(1) + not_a_template(0);
}
