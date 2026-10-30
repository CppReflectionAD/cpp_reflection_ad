// §3.4: return_operand_of on `return {a, b};` gives the resolved constructor
// call.
#include "probe.h"
struct P {
  P(double, double);
  double a, b;
};
P f(double x, double y) { return {x, y}; }
constexpr m::info e = m::return_operand_of(stmt<0>(^^f));
static_assert(m::is_expression(e), "P12 return {x, y} operand is_expression");
static_assert(m::operands_of(e).size() == 2,
              "P12 constructor call has 2 operands");
