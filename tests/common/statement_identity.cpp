// A reflection of a statement is equal to a reflection of the same statement,
// and only to that, on both compilers: two statements that look alike are
// still two statements, and a reflection stays equal to a fresh one when a
// constexpr variable or a class member holds it, or when it is passed to or
// returned from a consteval function.
//
// On GCC, comparing the reflections of two different null statements was an
// ICE (#282), fixed in CppReflectionAD/gcc#43; and the constant evaluator
// copied the statement whenever it copied a reflection, so a reflection of an
// `if constexpr` stopped being equal to itself once stored or passed (#283),
// fixed in CppReflectionAD/gcc#44. Both have more checks in
// g++.dg/reflect/null_statement4.C and statement_identity1.C. GCC compares a
// plain `if` by identity too since CppReflectionAD/gcc#39.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int two_nulls(int x) {
  ;
  ;
  return x;
}

inline int two_ifs(int x) {
  if (x)
    sink(1);
  if (x)
    sink(1);
  return 0;
}

template <int N> int two_if_constexprs() {
  if constexpr (N)
    sink(1);
  if constexpr (N)
    sink(1);
  return 0;
}

inline int block(int x) {
  {
    sink(x);
  }
  return 0;
}

consteval info st(info fn, std::size_t k) {
  return m::statements_of(m::body_of(fn))[k];
}
consteval bool same(info a, info b) { return a == b; }
consteval info id(info r) { return r; }

struct holder {
  info a;
  info b;
};

// Two statements that look alike are two statements (#282 for the null ones).
static_assert(st(^^two_nulls, 0) == st(^^two_nulls, 0), "a null statement");
static_assert(st(^^two_nulls, 0) != st(^^two_nulls, 1), "two null statements");
static_assert(st(^^two_ifs, 0) != st(^^two_ifs, 1), "two `if`s");
static_assert(st(^^two_if_constexprs<1>, 0) != st(^^two_if_constexprs<1>, 1),
              "two `if constexpr`s");

// A stored or passed reflection is still equal to a fresh one (#283).
#define CHECK_KEPT(NAME, FN, K)                                                \
  constexpr info NAME = st(FN, K);                                             \
  static_assert(NAME == st(FN, K), #NAME ": a constexpr variable");            \
  static_assert(id(NAME) == NAME, #NAME ": passed and returned");              \
  static_assert(same(st(FN, K), st(FN, K)), #NAME ": two arguments");          \
  static_assert(holder{st(FN, K), st(FN, K)}.a == st(FN, K),                   \
                #NAME ": a class member");                                     \
  static_assert(id(NAME) != st(FN, (K) + 1), #NAME ": the next differs");

CHECK_KEPT(kept_null, ^^two_nulls, 0)
CHECK_KEPT(kept_if, ^^two_ifs, 0)
CHECK_KEPT(kept_if_constexpr, ^^two_if_constexprs<1>, 0)
CHECK_KEPT(kept_block, ^^block, 0)

#undef CHECK_KEPT

constexpr holder both{st(^^two_if_constexprs<1>, 0),
                      st(^^two_if_constexprs<1>, 1)};
static_assert(both.a == kept_if_constexpr, "a constexpr class member");
static_assert(both.a != both.b, "two constexpr class members");

int main() {
  return two_nulls(0) + two_ifs(0) + two_if_constexprs<1>() + block(0);
}
