// Known GCC gap: a function body using a class with a non-trivial destructor
// loses its `return` statement.
//
// With a trivially destructible type, both compilers report the `return` in
// these bodies. Give the type a destructor and GCC's statements_of() no longer
// contains a return statement: after a local of that type
// (local_then_return), and even when the body is just `return make(x);`
// (return_call). Clang reports both returns, and the returned `make(x)` as a
// function call. This blocks reflecting code over heap-backed types such as a
// tensor with std::vector storage.
//
// tests/CMakeLists.txt expects both static_asserts below to fail, with the
// errors in nontrivial_destructor_reflection.fail.txt, so the test also fails
// if GCC fixes only one of them. clang checks the same asserts in
// tests/clang_only/nontrivial_destructor_reflection.cpp. When GCC fixes the
// gap, move that file to tests/common/ and delete this one and its .fail.txt.
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
