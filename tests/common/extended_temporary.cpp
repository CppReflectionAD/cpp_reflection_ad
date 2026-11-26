// A declaration that extends the lifetime of a temporary is one declaration
// statement on both compilers ([class.temporary]), whether or not the
// temporary's type has a destructor, and so is one whose initializer is a
// std::initializer_list. A label before it labels the declaration, and the
// init-statement of a `for` that declares one is the declaration.
//
// GCC used to give the variable that holds the temporary, which it declares
// ahead of the declaration, as a statement of its own before it (#293). Fixed
// in CppReflectionAD/gcc#49, which has more checks in
// g++.dg/reflect/extended_temp2.C.
#include <ad/meta.h>
#include <cstddef>
#include <initializer_list>
namespace m = ad::meta;
using std::meta::info;
struct T {
  ~T() {}
  int v = 1;
};
struct U {
  int v = 1;
};
struct A {
  const T &t;
};
inline int sink(int x) { return x; }

inline int with_dtor(int x) {
  const T &r = T{};
  return x + r.v;
}
inline int trivial(int x) {
  const U &r = U{};
  return x + r.v;
}
inline int scalar(int x) {
  const int &i = 5;
  return x + i;
}
inline int rvalue(int x) {
  T &&r = T{};
  return x + r.v;
}
inline int static_local(int x) {
  static const T &s = T{};
  return x + s.v;
}
inline int init_list(int x) {
  std::initializer_list<int> l = {1, 2, 3};
  return x + static_cast<int>(l.size());
}
inline int aggregate(int x) {
  A a{T{}};
  return x + a.t.v;
}
inline int conditional(int x) {
  const T &r = x ? T{} : T{};
  return r.v;
}
inline int two(int x) {
  const T &a = T{}, &b = T{};
  return x + a.v + b.v;
}
inline int labeled(int x) {
[[maybe_unused]] l:
  const T &r = T{};
  return x + r.v;
}
inline int nested(int x) {
  {
    const T &r = T{};
    sink(r.v);
  }
  return x;
}
inline int for_init(int x) {
  for (const T &r = T{}; x; --x)
    sink(r.v);
  return 0;
}
inline int for_init_trivial(int x) {
  for (const U &r = U{}; x; --x)
    sink(r.v);
  return 0;
}
template <typename X> X in_template(X x) {
  const T &r = T{};
  return x + r.v;
}

consteval std::size_t parts(info s) { return m::statements_of(s).size(); }

consteval info part(info s, std::size_t i) { return m::statements_of(s)[i]; }

// Statement i of fn's body.
consteval info st(info fn, std::size_t i) { return part(m::body_of(fn), i); }

// fn's body is a declaration with one initializer, then a return.
consteval bool declares_then_returns(info fn) {
  return parts(m::body_of(fn)) == 2 && m::is_declaration_statement(st(fn, 0)) &&
         m::initializers_of(st(fn, 0)).size() == 1 &&
         m::is_return_statement(st(fn, 1));
}

static_assert(declares_then_returns(^^with_dtor), "with a destructor");
static_assert(declares_then_returns(^^trivial), "with none");
static_assert(declares_then_returns(^^scalar), "a scalar");
static_assert(declares_then_returns(^^rvalue), "an rvalue reference");
static_assert(declares_then_returns(^^static_local), "a static reference");
static_assert(declares_then_returns(^^init_list), "a std::initializer_list");
static_assert(declares_then_returns(^^aggregate), "a reference member");
static_assert(declares_then_returns(^^conditional), "a conditional");
static_assert(declares_then_returns(^^in_template<int>), "in a template");
static_assert(m::is_expression(m::initializers_of(st(^^with_dtor, 0))[0]),
              "the initializer is an expression");

static_assert(parts(m::body_of(^^two)) == 2 &&
                  m::is_declaration_statement(st(^^two, 0)) &&
                  m::initializers_of(st(^^two, 0)).size() == 2,
              "one declaration of two references");

static_assert(parts(m::body_of(^^labeled)) == 2 &&
                  parts(st(^^labeled, 0)) == 1 &&
                  m::is_declaration_statement(part(st(^^labeled, 0), 0)),
              "the label labels the declaration");

static_assert(parts(st(^^nested, 0)) == 2 &&
                  m::is_declaration_statement(part(st(^^nested, 0), 0)) &&
                  m::is_expression_statement(part(st(^^nested, 0), 1)),
              "in a nested block");

// The init-statement of a `for` that declares such a reference.
consteval bool declares_init(info fn) {
  info init = m::init_statement_of(st(fn, 0));
  return m::is_declaration_statement(init) &&
         m::initializers_of(init).size() == 1;
}
static_assert(declares_init(^^for_init), "a for, with a destructor");
static_assert(declares_init(^^for_init_trivial), "a for, with none");

int main() {
  return with_dtor(0) + trivial(0) + scalar(0) + rvalue(0) + static_local(0) +
         init_list(0) + aggregate(0) + conditional(0) + two(0) + labeled(0) +
         nested(0) + for_init(0) + for_init_trivial(0) + in_template(0) - 18;
}
