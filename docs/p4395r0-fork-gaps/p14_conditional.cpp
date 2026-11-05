// §3.5 (verbatim example) + §2.3 + §3.6: a conditional expression has three
// operands; the paper proposes extending std::meta::operators with it.
#include "probe.h"
using namespace std::meta;
int max(int x, int y) { return (x > y) ? x : y; }
constexpr info e = return_operand_of(statements_of(body_of(^^max))[0]);
static_assert(is_expression(e), "P14 is_expression");
static_assert(operands_of(e).size() == 3, "P14 operands_of(?:) size 3");
static_assert(is_expression(operands_of(e)[0]), "P14 condition is_expression");

bool overdue(int days_open, bool member) {
  return days_open > (member ? 60 : 30);
}
constexpr info g = return_operand_of(statements_of(body_of(^^overdue))[0]);
static_assert(operands_of(operands_of(g)[1]).size() == 3,
              "P14 nested ?: has 3 operands");
