// A function body using a class with a non-trivial destructor keeps its
// `return` statement, and a returned call is a function call.
//
// clang only, because GCC loses the return statements: see
// tests/known_gap/gcc/nontrivial_destructor_reflection.cpp, which holds the
// same asserts. When GCC fixes the gap, move this file to tests/common/ and
// delete that one.
#include <meta>

namespace m = std::meta;

struct D {
  double v;
  ~D() {}
};

D make(double x) { return {x}; }

double local_then_return(double x) {
  D a = make(x);
  return a.v;
}

D return_call(double x) { return make(x); }

consteval bool has_return(m::info fn) {
  for (m::info s : m::statements_of(m::body_of(fn)))
    if (m::is_return_statement(s))
      return true;
  return false;
}

// The function returns a call, once implicit conversions are peeled off.
consteval bool returns_a_call(m::info fn) {
  for (m::info s : m::statements_of(m::body_of(fn)))
    if (m::is_return_statement(s)) {
      m::info e = m::return_value_of(s);
      while (m::is_cast(e))
        e = m::operands_of(e).front();
      return m::is_function_call(e);
    }
  return false;
}

static_assert(has_return(^^local_then_return),
              "no return statement after a local with a destructor");
static_assert(returns_a_call(^^return_call),
              "`return make(x);` is not a return of a function call");

int main() {}
