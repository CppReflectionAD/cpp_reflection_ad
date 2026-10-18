#ifndef REFLECTION_RETURN_EXPR_HPP
#define REFLECTION_RETURN_EXPR_HPP

#include <meta>

// The first top-level `return` statement in a function's body. Throws, rather
// than returning a null reflection, if there is none, so a static_assert that
// uses it fails with "no return statement" instead of a misleading message.
consteval std::meta::info return_statement_of(std::meta::info fn) {
  for (std::meta::info s : std::meta::statements_of(std::meta::body_of(fn)))
    if (std::meta::is_return_statement(s))
      return s;
  throw "no return statement";
}

// The expression in a function's `return` statement, with implicit
// conversions peeled off.
consteval std::meta::info return_expr(std::meta::info fn) {
  std::meta::info e = std::meta::return_operand_of(return_statement_of(fn));
  while (std::meta::is_conversion(e))
    e = std::meta::operands_of(e).front();
  return e;
}

#endif
