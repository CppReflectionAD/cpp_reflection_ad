#ifndef REFLECTION_RETURN_EXPR_HPP
#define REFLECTION_RETURN_EXPR_HPP

#include <ad/meta.h>

// The first top-level `return` statement in a function's body. Throws, rather
// than returning a null reflection, if there is none, so a static_assert that
// uses it fails with "no return statement" instead of a misleading message.
consteval ad::meta::info return_statement_of(ad::meta::info fn) {
  for (ad::meta::info s : ad::meta::statements_of(ad::meta::body_of(fn)))
    if (ad::meta::is_return_statement(s))
      return s;
  throw "no return statement";
}

// The expression in a function's `return` statement, with implicit
// conversions peeled off.
consteval ad::meta::info return_expr(ad::meta::info fn) {
  ad::meta::info e = ad::meta::return_operand_of(return_statement_of(fn));
  while (ad::meta::is_conversion(e))
    e = ad::meta::operands_of(e).front();
  return e;
}

#endif
