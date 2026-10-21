#ifndef REFLECTION_AD_TESTING_GAP_HPP
#define REFLECTION_AD_TESTING_GAP_HPP

// Stand-ins for two expression-reflection metafunctions that P4395 proposes
// but the clang fork doesn't provide yet. Each is built on the fork's private
// API (__metafunction and its expression kind numbers), which can change at
// any clang-p2996 pin bump; the static_assert at the end makes such a change a
// build error rather than wrong output. Call them qualified (`gap::`): an
// unqualified call also finds std::meta's by argument-dependent lookup.
//
// The missing metafunctions are pinned as known gaps in
// tests/known_gap/clang/. When the fork provides one, its test there starts
// failing: replace `gap::<name>` with `std::meta::<name>` where it's used and
// delete the stand-in here. When both are gone, delete this header.

#include <cstddef>
#include <meta>

namespace reflect_test::gap {
namespace m = std::meta;

// std::meta::expression_of (P4395 §3.5): the expression of an expression
// statement. The fork has the metafunction but no public wrapper for it.
// Known gap: tests/known_gap/clang/public_expression_of.cpp.
consteval m::info expression_of(m::info statement) {
  return __metafunction(m::detail::__metafn_expression_of, statement);
}

// The fork's private kind number for a built-in subscript `a[i]`.
inline constexpr std::size_t builtin_subscript_kind = 9;

// std::meta::expression_operator_of (P4395 §3.5, §3.6), which should give
// op_square_brackets for a built-in subscript `a[i]`. The fork's only maps
// built-in unary and binary operators, so it gives no operator for `a[i]`,
// which is recognised here by its private kind number instead. The fork fix
// is one line in its expression_operator_of
// (clang/lib/AST/ExprConstantMeta.cpp): map ArraySubscriptExpr to
// OO_Subscript.
// Known gap: tests/known_gap/clang/builtin_subscript_operator.cpp.
consteval m::operators expression_operator_of(m::info e) {
  if (__metafunction(m::detail::__metafn_expression_kind_of, e) ==
      builtin_subscript_kind)
    return m::op_square_brackets;
  return m::expression_operator_of(e);
}

namespace detail {
inline void probe(int *p) { p[0] = 1; }

// Whether the stand-ins still work: on probe()'s body, expression_of gives
// the assignment and expression_operator_of recognises its subscript.
consteval bool stand_ins_work() {
  const m::info s = m::statements_of(m::body_of(^^probe))[0];
  const m::info e = gap::expression_of(s);
  if (!m::is_binary_operator(e))
    return false;
  m::info lhs = m::operands_of(e)[0];
  while (m::is_cast(lhs))
    lhs = m::operands_of(lhs)[0];
  return gap::expression_operator_of(lhs) == m::op_square_brackets;
}
} // namespace detail

static_assert(detail::stand_ins_work(),
              "the clang fork's private expression API changed; update "
              "include/testing/clang_only/gap.hpp");

} // namespace reflect_test::gap

#endif // REFLECTION_AD_TESTING_GAP_HPP
