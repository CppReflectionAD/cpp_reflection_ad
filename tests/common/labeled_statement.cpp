// A labeled statement is one statement whose last part is the statement it
// labels ([stmt.label]), on both compilers: `L: s` has the part `s`,
// `case v: s` the parts `v` and `s`, and `default: s` the part `s`. A label at
// the end of a block labels a null statement, and `a: b: s` nests.
//
// GCC used to give a label and the statement it labels as two statements of the
// block, and a variable with a destructor declared after a label as a block of
// its own (#287). Fixed in CppReflectionAD/gcc#48, which has more checks in
// g++.dg/reflect/labeled_statement1.C.
//
// Not checked here: GCC gives the value of a `case` label as an expression,
// and clang as an expression statement (#288). The engine still rejects a
// labeled statement in a body it differentiates.
#include <ad/meta.h>

#include <cstddef>

namespace m = ad::meta;
using std::meta::info;

struct T {
  ~T() {}
};

inline int sink(int x) { return x; }

inline int named(int x) {
[[maybe_unused]] a:
  sink(x);
[[maybe_unused]] b:
[[maybe_unused]] c:
  sink(x);
  return 0;
}

inline int at_end(int x) {
  {
    sink(x);
  [[maybe_unused]] e:
  }
  return 0;
}

inline int cases(int x) {
  switch (x) {
  case 1:
    sink(x);
    break;
  case 2:
  case 3:
    break;
  default:
    break;
  }
  return 0;
}

inline int class_after_label(int x) {
[[maybe_unused]] k:
  sink(x);
  T t;
  sink(x);
  return 0;
}

// clang-format off
inline int clauses(int x) {
  if (x)
    [[maybe_unused]] l1: sink(x);
  while (x)
    [[maybe_unused]] l2: sink(x--);
  return 0;
}
// clang-format on

template <typename U> U in_template(U x) {
[[maybe_unused]] l:
  T t;
  sink(x);
  return x;
}

consteval std::size_t parts(info s) { return m::statements_of(s).size(); }

consteval info part(info s, std::size_t i) { return m::statements_of(s)[i]; }

// Statement i of fn's body.
consteval info st(info fn, std::size_t i) { return part(m::body_of(fn), i); }

// A labeled statement with n parts, the last the statement it labels: one for a
// named label or `default:`, two for a `case` label.
consteval bool labels(info s, std::size_t n = 1) {
  return m::is_statement(s) && !m::is_compound_statement(s) &&
         !m::is_expression_statement(s) && parts(s) == n;
}

static_assert(parts(m::body_of(^^named)) == 3, "a:, b:, return");
static_assert(labels(st(^^named, 0)) &&
                  m::is_expression_statement(part(st(^^named, 0), 0)),
              "a: labels the expression statement after it");
static_assert(labels(st(^^named, 1)) && labels(part(st(^^named, 1), 0)) &&
                  m::is_expression_statement(part(part(st(^^named, 1), 0), 0)),
              "b: c: nests");
static_assert(m::is_return_statement(st(^^named, 2)), "then the return");

static_assert(parts(st(^^at_end, 0)) == 2 && labels(part(st(^^at_end, 0), 1)),
              "e: at the end of a block is a statement of the block");
static_assert(parts(part(part(st(^^at_end, 0), 1), 0)) == 0,
              "that labels a null statement");

// The block that is the body of the `switch` in cases.
consteval info switch_body(std::size_t i) {
  return part(part(st(^^cases, 0), 1), i);
}
static_assert(parts(part(st(^^cases, 0), 1)) == 4,
              "case 1:, break, case 2:, default:");
static_assert(labels(switch_body(0), 2) &&
                  m::is_expression_statement(part(switch_body(0), 1)),
              "case 1: has its value and the statement it labels");
static_assert(labels(switch_body(2), 2) && labels(part(switch_body(2), 1), 2),
              "case 2: labels case 3:");
static_assert(labels(switch_body(3)), "default: has the statement it labels");

static_assert(parts(m::body_of(^^class_after_label)) == 4 &&
                  m::is_declaration_statement(st(^^class_after_label, 1)),
              "a variable with a destructor after a label isn't in a block");

static_assert(labels(m::then_statement_of(st(^^clauses, 0))),
              "a then statement can be a labeled statement");
static_assert(labels(m::loop_body_of(st(^^clauses, 1))),
              "a loop body can be a labeled statement");

constexpr info templ = ^^in_template<int>;
static_assert(parts(m::body_of(templ)) == 3 &&
                  m::is_declaration_statement(part(st(templ, 0), 0)),
              "in a template");

int main() {
  return named(0) + at_end(0) + cases(0) + class_after_label(0) + clauses(0) +
         in_template(0);
}
