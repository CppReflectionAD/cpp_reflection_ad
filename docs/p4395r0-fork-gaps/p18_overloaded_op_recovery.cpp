// Can the operator of an overloaded `a + b` be recovered from the callee's
// declaration (P2996 operator_of on the operator function)?
#include "probe.h"
struct S {
  double v;
};
S operator+(S, S);
S f(S a, S b) { return a + b; }
constexpr m::info e = m::return_operand_of(stmt<0>(^^f));
constexpr m::info fn = m::declaration_of(m::callee_of(e));
static_assert(fn == ^^operator+, "P18 declaration_of(callee_of) is operator+");
static_assert(std::meta::is_operator_function(fn), "P18 is_operator_function");
static_assert(std::meta::operator_of(fn) == std::meta::operators::op_plus,
              "P18 operator_of == op_plus");
