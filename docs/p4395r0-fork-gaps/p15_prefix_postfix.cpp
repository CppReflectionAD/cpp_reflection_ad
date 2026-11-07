// §3.6: is_prefix / is_postfix distinguish ++a from a++.
#include "probe.h"
void f(int &i) {
  ++i;
  i++;
}
constexpr m::info pre = m::expression_of(stmt<0>(^^f));
constexpr m::info post = m::expression_of(stmt<1>(^^f));
static_assert(m::expression_operator_of(pre) == m::operators::op_plus_plus,
              "P15 ++i op_plus_plus");
static_assert(m::expression_operator_of(post) == m::operators::op_plus_plus,
              "P15 i++ op_plus_plus");
static_assert(m::is_prefix(pre) && !m::is_postfix(pre), "P15 ++i is_prefix");
static_assert(m::is_postfix(post) && !m::is_prefix(post), "P15 i++ is_postfix");
