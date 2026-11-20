// A plain `if` with a constant condition is one statement in the block it is
// in, on both compilers, whichever of its statements the condition selects
// (P4395R0 §3.2).
//
// GCC used to replace it by the statement its condition selects, where the
// other statement has no side effects, so statements_of gave that statement,
// or that statement's statements, or nothing for `if (false) {}` (#268).
// Fixed in CppReflectionAD/gcc#33, which has these checks, and more, in
// g++.dg/reflect/if_constant1.C. static_fail/constant_if.cpp checks that the
// engine now rejects such an `if` on both compilers, as it does any `if`.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int then_block(int x) {
  if (true) {
    sink(x);
  }
  return 0;
}

inline int two_in_block(int x) {
  if (sizeof(int) >= 2) {
    sink(x);
    sink(1);
  }
  return 0;
}

inline int pure_else(int x) {
  if (true)
    sink(x);
  else
    (void)x;
  return 0;
}

inline int false_empty(int x) {
  if (false) {
  }
  return x;
}

inline int false_else(int x) {
  if (false) {
  } else
    sink(x);
  return 0;
}

template <typename T> int in_template(T x) {
  if (sizeof(T) >= 2)
    return static_cast<int>(x);
  return 0;
}

inline int not_constant(int x) {
  if (x)
    sink(x);
  return 0;
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

// The `if`, then the return.
consteval bool if_then_return(info fn) {
  return count(fn) == 2 && other_statement(first(fn)) &&
         m::is_return_statement(m::statements_of(m::body_of(fn))[1]);
}

static_assert(if_then_return(^^then_block), "a one-statement block");
static_assert(if_then_return(^^two_in_block), "a two-statement block");
static_assert(if_then_return(^^pure_else), "an else with no side effects");
static_assert(if_then_return(^^false_empty), "nothing selected");
static_assert(if_then_return(^^false_else), "the else selected");
static_assert(if_then_return(^^in_template<int>), "in a template");
static_assert(if_then_return(^^not_constant),
              "a condition that isn't constant");

int main() {
  return then_block(1) + two_in_block(1) + pure_else(1) + false_empty(0) +
         false_else(1) + in_template(0) + not_constant(0);
}
