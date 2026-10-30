// §3.2/§4.2: reflecting an instantiation drops the discarded `if constexpr`
// branch. The paper's example body has one statement (the if); the discarded
// else must not be reachable.
#include "probe.h"
#include <type_traits>
int sink(int);
template <typename T> int doSomething(T) {
  if constexpr (std::is_arithmetic_v<T>) {
    sink(1);
  } else {
    sink(2);
    sink(3);
  }
  return 0;
}
static_assert(nstmts(^^doSomething<int>) == 2,
              "P03 body has 2 statements (if, return)");
static_assert(m::is_statement(stmt<0>(^^doSomething<int>)),
              "P03 first is a statement");
#ifdef __clang__
static_assert(m::is_if_statement(stmt<0>(^^doSomething<int>)),
              "P03 clang: first is still an if");
static_assert(m::else_statement_of(stmt<0>(^^doSomething<int>)) == m::info{},
              "P03 clang: discarded else is absent");
#endif
