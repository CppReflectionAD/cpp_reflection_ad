// meta.h — std::meta with the clang fork's expression-reflection API on both
// compilers.
//
// The clang fork renamed and reshaped part of the statement/expression API
// (siqiyyyy/clang-p2996#11); the GCC fork's <meta> still has the older one.
// Code that reflects function bodies uses ad::meta instead of std::meta, and
// on GCC the functions below supply the clang fork's API on top of the older
// one, with the same meaning:
//   - is_id_expression, is_conversion, return_operand_of: renames or
//     combinations of the old predicates.
//   - is_operator_expression: also true for `&&`, `||` and `!`, which GCC's
//     is_unary_operator and is_binary_operator leave out. Unlike clang's, it
//     is false for an overloaded operator, which GCC reflects only as a call.
//   - operands_of: on a call, the arguments only. GCC lists the callee first,
//     if the call has one.
//   - declaration_of: also accepts what GCC's callee_of returns.
//   - declarators_of / initializers_of: GCC reflects one variable per
//     declaration statement, so these have one element.
// On clang, ad::meta is std::meta. Delete the GCC part once the GCC fork has
// the same API; the static_assert below says when.

#ifndef REFLECT_DEMO_META_H
#define REFLECT_DEMO_META_H

#include <meta>
#include <type_traits>
#include <vector>

namespace ad::meta {
using namespace std::meta;

#if defined(_GLIBCXX_META)
consteval bool is_id_expression(info e) {
  return std::meta::is_variable_reference(e);
}

// GCC's expression_operator_of and operands_of already handle the logical
// operators; only the predicates leave them out.
consteval bool is_operator_expression(info e) {
  if (std::meta::is_unary_operator(e) || std::meta::is_binary_operator(e))
    return true;
  try {
    const auto op = std::meta::expression_operator_of(e);
    return op == std::meta::operators::op_ampersand_ampersand ||
           op == std::meta::operators::op_pipe_pipe ||
           op == std::meta::operators::op_exclamation;
  } catch (const std::meta::exception &) {
    return false;
  }
}

namespace detail {
// Whether the first of a call's GCC operands is its callee. A call to an
// internal function (e.g. __builtin_launder) has none, so its first operand
// is an argument, which never has function or member function pointer type.
// A virtual call's callee has no type at all.
consteval bool is_callee(info op) {
  if (std::meta::is_function(op))
    return true;
  info t;
  try {
    t = std::meta::remove_reference(std::meta::type_of(op));
  } catch (const std::meta::exception &) {
    return true;
  }
  if (std::meta::is_pointer_type(t))
    t = std::meta::remove_pointer(t);
  return std::meta::is_function_type(t) ||
         std::meta::is_member_function_pointer_type(t);
}
} // namespace detail

consteval std::vector<info> operands_of(info e) {
  std::vector<info> ops = std::meta::operands_of(e);
  if (std::meta::is_function_call(e) && !ops.empty() &&
      detail::is_callee(ops.front()))
    ops.erase(ops.begin());
  return ops;
}

// GCC's callee_of is the function itself, not an expression naming it. For
// std::max/min, GCC reflects the `const T&` return as `*call`, so callee_of
// gives the call; pass it through too, so it matches no callee, as before.
consteval info declaration_of(info e) {
  if (std::meta::is_function(e) || std::meta::is_function_call(e))
    return e;
  return std::meta::declaration_of(e);
}

consteval bool is_conversion(info e) {
  return std::meta::is_cast(e) ||
         (std::meta::is_construct(e) && ad::meta::operands_of(e).size() == 1);
}

consteval info return_operand_of(info s) {
  return std::meta::return_value_of(s);
}

consteval std::vector<info> declarators_of(info s) {
  return {std::meta::declared_variable_of(s)};
}

consteval std::vector<info> initializers_of(info s) {
  return {std::meta::initializer_of(std::meta::declared_variable_of(s))};
}

// Fails once GCC's <meta> has return_operand_of, as a sign that it has the
// rest of the new API too. Then the wrappers above would shadow it (and
// operands_of would drop a real argument), so delete them.
namespace detail {
struct absent {};
namespace fallback {
consteval absent return_operand_of(auto) { return {}; }
} // namespace fallback
namespace probe {
using namespace std::meta;
using namespace fallback;
} // namespace probe
} // namespace detail
static_assert(std::is_same_v<decltype(detail::probe::return_operand_of(^^int)),
                             detail::absent>,
              "GCC's <meta> now has the new expression-reflection API: "
              "delete the GCC wrappers in ad/meta.h");
#endif
} // namespace ad::meta

#endif
