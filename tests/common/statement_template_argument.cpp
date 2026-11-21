// A reflection of a statement can be the argument of a specialization that is
// never emitted, on both compilers. A statement has no mangling, so emitting
// one is an error on both.
//
// On GCC, a null statement or an alias declaration was an ICE (#284), and a
// block made the specialization unusable (#285), fixed in
// CppReflectionAD/gcc#46, which has more checks in
// g++.dg/reflect/stmt_template_arg2.C and stmt_template_arg3.C.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int f(int x) {
  ;
  using T = int;
  {
    sink(x);
  }
  if (x)
    sink(1);
  sink(x);
  T y = x;
  return y;
}

consteval info st(std::size_t k) {
  return m::statements_of(m::body_of(^^f))[k];
}

template <info R> struct tag {
  using type = int;
};

#define CHECK_ARG(K, WHAT)                                                     \
  static_assert(__is_same(typename tag<st(K)>::type, int), WHAT);              \
  static_assert(__is_same(tag<st(K)>, tag<st(K)>), WHAT ", twice");

CHECK_ARG(0, "a null statement")
CHECK_ARG(1, "an alias declaration")
CHECK_ARG(2, "a block")
CHECK_ARG(3, "an `if`")
CHECK_ARG(4, "an expression statement")
CHECK_ARG(5, "a declaration")
CHECK_ARG(6, "a `return`")

#undef CHECK_ARG

static_assert(!__is_same(tag<st(2)>, tag<st(4)>), "two statements");

int main() { return f(0); }
