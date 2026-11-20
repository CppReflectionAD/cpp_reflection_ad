// A compound assignment `a op= b` is reflected as written on both compilers:
// its operator is `op=`, and its operands are `a` and `b`.
//
// GCC used to reflect it as `a = a op b`, an assignment of the binary
// operation (#246). Fixed in CppReflectionAD/gcc#27, which has the same checks
// in g++.dg/reflect/compound_assign1.C.
#include <ad/meta.h>

#include <string_view>

namespace m = ad::meta;
using std::meta::info;
using ops = std::meta::operators;

struct P {
  double x, y;
};

inline void add(double &a, double b) { a += b; }
inline void sub(double &a, double b) { a -= b; }
inline void mul(double &a, double b) { a *= b; }
inline void quot(double &a, double b) { a /= b; }
inline void plain(double &a, double b) { a = a + b; }
inline void narrow(int &a, double b) { a += b; }
inline void member(P &p) { p.x += p.y * 0.5; }
template <typename T> void tsub(T &a, T b) { a -= b; }

consteval info expr(info fn) {
  return m::expression_of(m::statements_of(m::body_of(fn))[0]);
}

consteval info strip(info e) {
  while (m::is_conversion(e))
    e = m::operands_of(e)[0];
  return e;
}

consteval bool names(info e, std::string_view name) {
  e = strip(e);
  return m::is_id_expression(e) &&
         std::meta::identifier_of(m::declaration_of(e)) == name;
}

consteval bool compound(info fn, ops op) {
  info e = expr(fn);
  auto operands = m::operands_of(e);
  return m::expression_operator_of(e) == op && operands.size() == 2 &&
         names(operands[0], "a") && names(operands[1], "b");
}

static_assert(compound(^^add, ops::op_plus_equals));
static_assert(compound(^^sub, ops::op_minus_equals));
static_assert(compound(^^mul, ops::op_star_equals));
static_assert(compound(^^quot, ops::op_slash_equals));
static_assert(compound(^^narrow, ops::op_plus_equals));
static_assert(compound(^^tsub<double>, ops::op_minus_equals));

// `a = a + b` is still an assignment of `a + b`.
static_assert(m::expression_operator_of(expr(^^plain)) == ops::op_equals);
static_assert(m::expression_operator_of(
                  strip(m::operands_of(expr(^^plain))[1])) == ops::op_plus);

// P4395R0 section 2.5.
static_assert(m::expression_operator_of(expr(^^member)) == ops::op_plus_equals);
static_assert(m::expression_operator_of(
                  strip(m::operands_of(expr(^^member))[1])) == ops::op_star);

int main() { return 0; }
