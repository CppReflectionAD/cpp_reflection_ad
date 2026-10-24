// meta.h — std::meta with the clang fork's expression-reflection API on both
// compilers.
//
// The clang fork renamed and reshaped part of the statement/expression API
// (siqiyyyy/clang-p2996#11); the GCC fork's <meta> has only part of it so far
// (return_operand_of, CppReflectionAD/gcc#4). Code that reflects function
// bodies uses ad::meta instead of std::meta, and on GCC the functions below
// supply the rest of the clang fork's API on top of the older one, with the
// same meaning:
//   - is_id_expression, is_operator_expression, is_conversion: renames or
//     combinations of the old predicates.
//   - operands_of: on a call, the arguments only. GCC lists the callee first,
//     if the call has one.
//   - is_declaration_statement: false for an assignment, which GCC also
//     counts as a declaration statement.
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

consteval bool is_operator_expression(info e) {
  return std::meta::is_unary_operator(e) || std::meta::is_binary_operator(e);
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

namespace detail {
consteval info initializer_or_null(info r) {
  try {
    return std::meta::initializer_of(r);
  } catch (const std::meta::exception &) {
    return {};
  }
}
} // namespace detail

// GCC also reports an assignment `a = e;` (or `a += e;`) as a declaration
// statement of `a` (#185). On the statement, initializer_of gives the value it
// stores; on the variable, the variable's initializer. They differ unless the
// assignment stores that same initializer again.
consteval bool is_declaration_statement(info s) {
  return std::meta::is_declaration_statement(s) &&
         detail::initializer_or_null(s) ==
             detail::initializer_or_null(std::meta::declared_variable_of(s));
}

consteval bool is_conversion(info e) {
  return std::meta::is_cast(e) ||
         (std::meta::is_construct(e) && ad::meta::operands_of(e).size() == 1);
}

consteval std::vector<info> declarators_of(info s) {
  return {std::meta::declared_variable_of(s)};
}

consteval std::vector<info> initializers_of(info s) {
  return {std::meta::initializer_of(std::meta::declared_variable_of(s))};
}

// Fails once GCC's <meta> has declarators_of, as a sign that it has the rest
// of the new API too. Then the wrappers above would shadow it (and
// operands_of would drop a real argument), so delete them.
namespace detail {
struct absent {};
namespace fallback {
consteval absent declarators_of(auto) { return {}; }
} // namespace fallback
namespace probe {
using namespace std::meta;
using namespace fallback;
} // namespace probe
} // namespace detail
static_assert(std::is_same_v<decltype(detail::probe::declarators_of(^^int)),
                             detail::absent>,
              "GCC's <meta> now has the new expression-reflection API: "
              "delete the GCC wrappers in ad/meta.h");
#endif
} // namespace ad::meta

#endif
