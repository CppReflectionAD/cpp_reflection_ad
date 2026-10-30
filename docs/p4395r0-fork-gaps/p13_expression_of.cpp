// §3.5 + §2.5: expression_of on an expression statement; compound assignment
// with member-access operands (the paper's HPC `update` example).
#include "probe.h"
struct Particle {
  double x, y, z;
};
void update(Particle &p) { p.x += p.y * 0.5; }
static_assert(nstmts(^^update) == 1, "P13 one statement");
static_assert(m::is_expression_statement(stmt<0>(^^update)),
              "P13 is_expression_statement");
constexpr m::info e = m::expression_of(stmt<0>(^^update));
static_assert(m::is_expression(e), "P13 expression_of is_expression");
static_assert(m::is_operator_expression(e), "P13 += is_operator_expression");
static_assert(m::expression_operator_of(e) == m::operators::op_plus_equals,
              "P13 op_plus_equals");
static_assert(m::operands_of(e).size() == 2, "P13 operands_of size 2");
static_assert(m::is_expression(m::operands_of(e)[0]), "P13 p.x is_expression");
