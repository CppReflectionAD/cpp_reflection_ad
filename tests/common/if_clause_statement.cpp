// A clause of an `if` that isn't a block is the statement itself, whatever
// the debug flags, on both compilers.
//
// With `-gstatement-frontiers`, the default with `-g` at `-O1` and above, GCC
// used to give such a clause as a compound statement of a debug marker and the
// statement (#279). Fixed in CppReflectionAD/gcc#40, which has more checks in
// g++.dg/reflect/clause_debug_markers1.C. tests/CMakeLists.txt builds this
// file with `-gstatement-frontiers` on GCC, so every configuration checks it.
//
// The engine still rejects any `if` in a body it differentiates.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int both(int x) {
  if (x)
    sink(x);
  else
    sink(1);
  return 0;
}

inline int else_if(int x) {
  if (x)
    sink(x);
  else if (x + 1)
    sink(1);
  else
    sink(2);
  return 0;
}

template <typename T> int in_template(T x) {
  if (x)
    sink(x);
  else
    sink(1);
  return 0;
}

// The `if` that is the first statement of fn's body.
consteval info first(info fn) { return m::statements_of(m::body_of(fn))[0]; }

consteval info part(info s, std::size_t k) { return m::statements_of(s)[k]; }

static_assert(m::is_expression_statement(part(first(^^both), 1)), "then");
static_assert(m::is_expression_statement(part(first(^^both), 2)), "else");
static_assert(m::is_expression_statement(part(first(^^in_template<int>), 1)),
              "in a template: then");
static_assert(m::is_expression_statement(part(first(^^in_template<int>), 2)),
              "in a template: else");

// The `else if` is an `if`, not a compound statement holding one.
static_assert(!m::is_compound_statement(part(first(^^else_if), 2)),
              "`else if`");
static_assert(m::statements_of(part(first(^^else_if), 2)).size() == 3,
              "`else if`: condition, then, else");
static_assert(m::is_expression_statement(part(part(first(^^else_if), 2), 1)),
              "`else if`: its then");

int main() { return both(0) + else_if(0) + in_template(0); }
