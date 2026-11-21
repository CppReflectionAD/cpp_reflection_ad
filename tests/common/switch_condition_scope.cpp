// A `switch` whose init-statement or condition declares a variable is the
// `switch`, with the declarations as its first parts ([stmt.switch]), on both
// compilers.
//
// GCC used to give the scope around such a `switch` as a compound statement of
// the declarations and the `switch` (#278), as it did for an `if`. Fixed in
// CppReflectionAD/gcc#47, which has more checks in
// g++.dg/reflect/switch_cond_scope1.C.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

inline int sink(int x) { return x; }

inline int plain(int x) {
  switch (x) {
  case 1:
    sink(x);
    break;
  }
  return 0;
}

inline int declares(int x) {
  switch (int y = sink(x)) {
  case 1:
    sink(y);
    break;
  }
  return 0;
}

inline int init_statement(int x) {
  switch (int y = sink(x); int z = y) {
  case 1:
    sink(z);
    break;
  }
  return 0;
}

template <typename T> int in_template(T x) {
  switch (T y = x) {
  case 1:
    sink(y);
    break;
  }
  return 0;
}

inline int only(int x) {
  switch (int y = sink(x)) {
  case 1:
    return y;
  default:
    return 0;
  }
}

consteval std::size_t parts(info s) { return m::statements_of(s).size(); }

// The `switch` that is the first statement of fn's body.
consteval info first(info fn) { return m::statements_of(m::body_of(fn))[0]; }

consteval bool is_switch_shape(info s, std::size_t n) {
  return m::is_statement(s) && !m::is_compound_statement(s) && parts(s) == n &&
         m::is_compound_statement(m::statements_of(s)[n - 1]);
}

static_assert(is_switch_shape(first(^^plain), 2), "condition, body");
static_assert(is_switch_shape(first(^^declares), 3),
              "declaration, condition, body");
static_assert(is_switch_shape(first(^^init_statement), 4),
              "init-statement, declaration, condition, body");
static_assert(is_switch_shape(first(^^in_template<int>), 3),
              "in a template: declaration, condition, body");
static_assert(parts(m::body_of(^^only)) == 1 &&
                  is_switch_shape(first(^^only), 3),
              "a body that is only such a `switch` holds one statement");

int main() {
  return plain(0) + declares(0) + init_statement(0) + in_template(0) + only(0);
}
