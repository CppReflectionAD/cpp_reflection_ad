// Known clang gap: no public std::meta::expression_of.
//
// P4395 §3.5 proposes expression_of(s), the expression of an expression
// statement `s`. The clang fork implements it (__metafn_expression_of) but its
// <meta> has no public wrapper, so reflect_test_expressions.hpp calls the
// private metafunction through gap::expression_of in
// include/testing/clang_only/gap.hpp.
//
// This file expects the error recorded in public_expression_of.fail.txt. When
// the fork adds std::meta::expression_of, move this file to
// tests/clang_only/, delete the .fail.txt, and replace gap::expression_of with
// std::meta::expression_of (deleting it from gap.hpp).
#include <meta>

namespace m = std::meta;

void assign(int &x) { x = 1; }

consteval bool expression_of_gives_the_assignment() {
  const m::info s = m::statements_of(m::body_of(^^assign))[0];
  return m::is_binary_operator(m::expression_of(s));
}

static_assert(expression_of_gives_the_assignment());

int main() {}
