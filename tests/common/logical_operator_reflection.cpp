// `||`, `&&` and `!` are operator expressions, with two, two and one operands.
//
// Clang reports is_operator_expression == true for them. GCC's
// is_unary_operator and is_binary_operator leave them out, but its
// expression_operator_of and operands_of handle them, and ad::meta's
// is_operator_expression uses that on GCC (#162 step 12 aligns the GCC
// fork's API). Without it, reflection AD on GCC took a condition such as
// `x != 0.0 && 1.0 / x > 0.5` for a constant and failed with "'x' is not a
// constant expression".
#include <ad/meta.h>

#include "reflection/return_expr.hpp"

namespace m = ad::meta;

bool f_cmp(double x) { return x > 0.0; }
bool f_or(double x) { return (x > 0.0) || (x > 1.0); }
bool f_and(double x) { return (x > 0.0) && (x > 1.0); }
bool f_not(double x) { return !(x > 0.0); }

constexpr m::info CMP = return_expr(^^f_cmp);
constexpr m::info OR = return_expr(^^f_or);
constexpr m::info AND = return_expr(^^f_and);
constexpr m::info NOT = return_expr(^^f_not);

static_assert(m::is_operator_expression(CMP) &&
              m::operands_of(CMP).size() == 2);

static_assert(m::is_operator_expression(OR) && m::operands_of(OR).size() == 2,
              "|| is not a binary operator");
static_assert(m::is_operator_expression(AND) && m::operands_of(AND).size() == 2,
              "&& is not a binary operator");
static_assert(m::is_operator_expression(NOT) && m::operands_of(NOT).size() == 1,
              "! is not a unary operator");

static_assert(m::expression_operator_of(OR) == m::operators::op_pipe_pipe);
static_assert(m::expression_operator_of(AND) ==
              m::operators::op_ampersand_ampersand);
static_assert(m::expression_operator_of(NOT) == m::operators::op_exclamation);

int main() {}
