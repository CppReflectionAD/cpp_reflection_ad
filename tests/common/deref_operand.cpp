// A dereference passed as an argument is `*` applied to the pointer, on both
// compilers, whatever the parameter's type.
//
// GCC used to report `*p` bound to a reference as a unary `+` applied to `p`
// (#258), which the engine treats as a no-op, and so accepted a dereference
// that clang rejects. Fixed in CppReflectionAD/gcc#26. The fork has the same
// checks, and more, in g++.dg/reflect/deref_operand1.C.
#include <ad/meta.h>

#include <string_view>

namespace m = ad::meta;

struct S {
  double v;
};
struct N {
  double v;
  N(const N &);
  ~N();
};

double ref_d(const double &);
double ref_s(const S &);
double ref_n(const N &);
double val_d(double);
double val_s(S);
double val_n(N);

// Bound to a reference.
inline double arg_d(const double *p) { return ref_d(*p); }
inline double arg_s(const S *p) { return ref_s(*p); }
inline double arg_n(const N *p) { return ref_n(*p); }
// Copied into a by-value parameter: trivially, and through a copy constructor.
inline double copy_s(const S *p) { return val_s(*p); }
inline double copy_n(const N *p) { return val_n(*p); }
// A scalar by value, which GCC always got right.
inline double scalar(const double *p) { return val_d(*p); }
// A written unary plus stays one.
inline double plus_d(double x) { return ref_d(+x); }

consteval m::info strip(m::info e) {
  while (m::is_conversion(e) && !m::operands_of(e).empty())
    e = m::operands_of(e)[0];
  return e;
}

consteval m::info argument(m::info fn) {
  auto call = m::return_operand_of(m::statements_of(m::body_of(fn))[0]);
  return strip(m::operands_of(call)[0]);
}

// E is `op NAME`, with nothing but conversions between the operator and NAME.
consteval bool applies(m::info e, m::operators op, std::string_view name) {
  if (!m::is_operator_expression(e) || m::expression_operator_of(e) != op)
    return false;
  auto ops = m::operands_of(e);
  return ops.size() == 1 && m::is_id_expression(strip(ops[0])) &&
         m::identifier_of(m::declaration_of(strip(ops[0]))) == name;
}

static_assert(applies(argument(^^arg_d), m::operators::op_star, "p"),
              "*p bound to const double &");
static_assert(applies(argument(^^arg_s), m::operators::op_star, "p"),
              "*p bound to a trivially copyable class reference");
static_assert(applies(argument(^^arg_n), m::operators::op_star, "p"),
              "*p bound to a non-trivial class reference");
static_assert(applies(argument(^^copy_s), m::operators::op_star, "p"),
              "*p copied into a trivially copyable class");
static_assert(applies(argument(^^copy_n), m::operators::op_star, "p"),
              "*p copied into a non-trivial class");
static_assert(applies(argument(^^scalar), m::operators::op_star, "p"),
              "*p passed as a double");
static_assert(applies(argument(^^plus_d), m::operators::op_plus, "x"),
              "+x bound to const double &");

int main() {}
