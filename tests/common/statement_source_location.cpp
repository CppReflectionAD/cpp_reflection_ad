// source_location_of a reflection of a statement gives the line where the
// statement starts, on both compilers, also when the reflection is stored in a
// constexpr variable or passed to a consteval function.
//
// GCC gave line 0 for every statement (#286), fixed in CppReflectionAD/gcc#45,
// which has more checks in g++.dg/reflect/source_location_statement1.C. A block
// still gives line 0 on GCC, which keeps only where a block ends, so it isn't
// checked here. Columns differ between the compilers, so only lines are.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int f(int x) {
  sink(x);
  if (x)
    sink(1);
  if (x)
    sink(1);
  else
    sink(2);
  ;
  int y = 0;
  using T = int;
  for (T i = 0; i < x; ++i)
    sink(i);
  while (x)
    break;
  do
    sink(3);
  while (false);
  switch (x) {
  case 1:
    break;
  }
  if constexpr (sizeof(T) > 0)
    sink(4);
  x = x + y;
  return x;
}

template <typename T> T g(T x) {
  sink(1);
  return x;
}

consteval info st(info fn, std::size_t k) {
  return m::statements_of(m::body_of(fn))[k];
}

// The line of R, counted from the line where FN's name is.
consteval int offset(info r, info fn) {
  return static_cast<int>(m::source_location_of(r).line()) -
         static_cast<int>(m::source_location_of(fn).line());
}
consteval int at(info fn, std::size_t k) { return offset(st(fn, k), fn); }

static_assert(m::source_location_of(st(^^f, 0)).line() != 0,
              "a statement has a line");
static_assert(at(^^f, 0) == 1, "an expression statement");
static_assert(at(^^f, 1) == 2, "an `if`");
static_assert(at(^^f, 2) == 4, "an `if` with `else`");
static_assert(at(^^f, 3) == 8, "a null statement");
static_assert(at(^^f, 4) == 9, "a declaration");
static_assert(at(^^f, 5) == 10, "an alias declaration");
static_assert(at(^^f, 6) == 11, "a `for`");
static_assert(at(^^f, 7) == 13, "a `while`");
static_assert(at(^^f, 8) == 15, "a `do`");
static_assert(at(^^f, 9) == 18, "a `switch`");
static_assert(at(^^f, 10) == 22, "an `if constexpr`");
static_assert(at(^^f, 11) == 24, "an assignment");
static_assert(at(^^f, 12) == 25, "a `return`");

static_assert(offset(m::statements_of(st(^^f, 2))[2], ^^f) == 7,
              "the `else` clause");
static_assert(at(^^g<int>, 0) == 1 && at(^^g<int>, 1) == 2,
              "statements of a template instantiation");

constexpr info stored = st(^^f, 10);
static_assert(offset(stored, ^^f) == 22, "stored in a constexpr variable");

consteval int line_of(info r) {
  return static_cast<int>(m::source_location_of(r).line());
}
static_assert(line_of(st(^^f, 1)) == line_of(st(^^f, 0)) + 1,
              "passed to a consteval function");

int main() { return f(0) + g(0); }
