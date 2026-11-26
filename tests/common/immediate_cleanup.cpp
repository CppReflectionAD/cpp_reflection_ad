// The body of a consteval function is reflected as that of any other
// function: after a local with a destructor, `return t.v;` is a return
// statement, on both compilers.
//
// GCC never genericizes an immediate function, so it keeps the local's
// CLEANUP_STMT, and statements_of didn't look through it: the statements
// after the local were one expression (#299). Fixed in
// CppReflectionAD/gcc#50; the fork's checks are in
// g++.dg/reflect/immediate_cleanup1.C.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;

struct T {
  constexpr ~T() {}
  int v = 1;
};

consteval int ce_local(int x) {
  T t;
  return t.v + x;
}
constexpr int cx_local(int x) {
  T t;
  return t.v + x;
}

consteval int ce_two(int x) {
  T a;
  T b;
  return a.v + b.v + x;
}

consteval int ce_in_if(int x) {
  if (x) {
    T t;
    return t.v;
  }
  return 0;
}

// Whether fn's body is n declarations and a return.
consteval bool declarations_then_return(m::info fn, std::size_t n) {
  auto stmts = m::statements_of(m::body_of(fn));
  if (stmts.size() != n + 1 || !m::is_return_statement(stmts[n]))
    return false;
  for (std::size_t i = 0; i < n; ++i)
    if (!m::is_declaration_statement(stmts[i]))
      return false;
  return true;
}

static_assert(declarations_then_return(^^cx_local, 1));
static_assert(declarations_then_return(^^ce_local, 1),
              "in a consteval function, the statements after a local with a "
              "destructor are not reflected as statements");
static_assert(declarations_then_return(^^ce_two, 2));

// The `if`'s block: a declaration and a return.
consteval bool if_block_returns(m::info fn) {
  auto stmts = m::statements_of(m::body_of(fn));
  if (stmts.size() != 2 || !m::is_if_statement(stmts[0]))
    return false;
  auto block = m::statements_of(m::then_statement_of(stmts[0]));
  return block.size() == 2 && m::is_declaration_statement(block[0]) &&
         m::is_return_statement(block[1]);
}

static_assert(if_block_returns(^^ce_in_if));

static_assert(ce_local(1) == 2 && ce_two(1) == 3 && ce_in_if(1) == 1);

int main() { return cx_local(1) == 2 ? 0 : 1; }
