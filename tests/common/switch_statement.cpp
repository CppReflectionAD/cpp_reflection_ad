// A `switch` is a statement whose parts are its condition and its body
// ([stmt.switch]), on both compilers.
//
// GCC used to give a `switch` as an expression, with no parts (#280). Fixed in
// CppReflectionAD/gcc#37.
//
// Not checked here: GCC gives the condition as an expression, and clang as an
// expression statement (#288); GCC gives a `case` label and the statement it
// labels as two statements of the body, and clang as one (#287); and a
// `switch` that declares a variable is still a compound statement around it on
// GCC (#278). The engine still rejects any `switch` in a body it
// differentiates.
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
  default:
    break;
  }
  return 0;
}

template <typename T> int in_template(T x) {
  switch (x) {
  case 1:
    sink(x);
  }
  return 0;
}

// The `switch` that is the first statement of fn's body.
consteval info first(info fn) { return m::statements_of(m::body_of(fn))[0]; }

consteval std::size_t parts(info s) { return m::statements_of(s).size(); }

consteval bool is_switch_shape(info s) {
  return m::is_statement(s) && !m::is_expression(s) &&
         !m::is_compound_statement(s) && !m::is_expression_statement(s) &&
         parts(s) == 2 && m::is_compound_statement(m::statements_of(s)[1]);
}

static_assert(is_switch_shape(first(^^plain)),
              "a statement of a condition and a block");
static_assert(is_switch_shape(first(^^in_template<int>)), "in a template");

int main() { return plain(1) + in_template(1); }
