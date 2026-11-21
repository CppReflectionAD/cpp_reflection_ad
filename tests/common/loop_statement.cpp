// A `while`, `for` or `do` is a statement whose parts are its own: the
// condition and the body, with the init-statement and the increment of a
// `for` ([stmt.iter]), on both compilers.
//
// GCC used to give a loop as a compound statement of the pieces of its lowered
// form, not all of them statements (#247). Fixed in
// CppReflectionAD/gcc#37, which has more checks in
// g++.dg/reflect/loop_statement1.C.
//
// Not checked here: GCC gives the condition as an expression, and clang as an
// expression statement (#288). The engine still rejects any loop in a body it
// differentiates.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int while_loop(int n) {
  while (n)
    --n;
  return 0;
}

inline int while_block(int n) {
  while (n) {
    --n;
    sink(n);
  }
  return 0;
}

inline int for_loop(int n) {
  for (int i = 0; i < n; ++i)
    --n;
  return 0;
}

inline int for_ever(int n) {
  for (;;)
    if (n)
      break;
  return 0;
}

inline int do_loop(int n) {
  do
    --n;
  while (n);
  return 0;
}

template <typename T> int in_template(T n) {
  while (n)
    --n;
  return 0;
}

// The loop that is the first statement of fn's body.
consteval info first(info fn) { return m::statements_of(m::body_of(fn))[0]; }

consteval std::size_t parts(info s) { return m::statements_of(s).size(); }

consteval info part(info s, std::size_t k) { return m::statements_of(s)[k]; }

consteval bool is_loop(info s) {
  return m::is_statement(s) && !m::is_expression(s) &&
         !m::is_compound_statement(s);
}

static_assert(is_loop(first(^^while_loop)), "`while`");
static_assert(parts(first(^^while_loop)) == 2, "`while`: condition, body");
static_assert(m::is_expression_statement(part(first(^^while_loop), 1)),
              "`while`: the body");

static_assert(is_loop(first(^^while_block)), "`while` with a block");
static_assert(parts(first(^^while_block)) == 2, "`while` with a block");
static_assert(m::is_compound_statement(part(first(^^while_block), 1)),
              "`while`: the block");
static_assert(parts(part(first(^^while_block), 1)) == 2,
              "`while`: the block's statements");

static_assert(is_loop(first(^^for_loop)), "`for`");
static_assert(parts(first(^^for_loop)) == 4,
              "`for`: init-statement, condition, increment, body");
static_assert(m::is_expression_statement(part(first(^^for_loop), 3)),
              "`for`: the body");

static_assert(is_loop(first(^^for_ever)), "`for (;;)`");
static_assert(parts(first(^^for_ever)) == 1, "`for (;;)`: only the body");

static_assert(is_loop(first(^^do_loop)), "`do`");
static_assert(parts(first(^^do_loop)) == 2, "`do`: body, condition");
static_assert(m::is_expression_statement(part(first(^^do_loop), 0)),
              "`do`: the body comes first");

static_assert(is_loop(first(^^in_template<int>)), "in a template");
static_assert(parts(first(^^in_template<int>)) == 2, "in a template");

int main() {
  return while_loop(2) + while_block(2) + for_loop(2) + for_ever(1) +
         do_loop(2) + in_template(2);
}
