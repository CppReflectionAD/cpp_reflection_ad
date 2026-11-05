// §3.2/§4.2: reflecting an instantiation drops the discarded `if constexpr`
// branch. The if is one statement however many statements its taken branch has,
// and the discarded else must not be reachable. The paper's example has the if
// as the whole body.
#include "probe.h"
#include <type_traits>
int sink(int);
template <typename T> void whole_body(T) {
  if constexpr (std::is_arithmetic_v<T>) {
    sink(1);
    sink(2);
  } else {
    sink(3);
  }
}
template <typename T> int then_return(T) {
  if constexpr (std::is_arithmetic_v<T>) {
    sink(1);
    sink(2);
  } else {
    sink(3);
  }
  return 0;
}
template <typename T> void not_taken(T) {
  if constexpr (!std::is_arithmetic_v<T>) {
    sink(1);
  }
}
static_assert(nstmts(^^whole_body<int>) == 1, "P03 one statement (the if)");
static_assert(nstmts(^^then_return<int>) == 2,
              "P03 two statements (the if, the return)");
static_assert(nstmts(^^not_taken<int>) == 1,
              "P03 an if with no taken branch is still one statement");
#ifdef __clang__
static_assert(m::is_if_statement(stmt<0>(^^then_return<int>)),
              "P03 clang: first is still an if");
static_assert(m::else_statement_of(stmt<0>(^^then_return<int>)) == m::info{},
              "P03 clang: discarded else is absent");
#endif
