// meta.h — the namespace that code reflecting function bodies uses.
//
// ad::meta is std::meta. Both forks now have the same statement/expression
// API (siqiyyyy/clang-p2996#11, CppReflectionAD/gcc#11); ad::meta is where a
// workaround for one fork goes. On GCC it adds:
//   - is_if_statement, which the clang fork has and the GCC fork doesn't.
//     GCC reflects an `if` as the same node as `?:`, so std::meta's
//     condition_of, true_expression_of and false_expression_of already accept
//     one; an `if` without an else has a void conversion as its
//     false_expression_of, not a null reflection. GCC also lowers a loop to
//     labels, gotos and an `if (cond) goto body; else goto exit;` before
//     reflection sees it, and that is an if statement here too, so
//     condition_of on it gives the loop's condition. clang keeps the loop
//     (see docs/gcc-loop-lowering.md).

#ifndef REFLECT_DEMO_META_H
#define REFLECT_DEMO_META_H

#include <meta>

namespace ad::meta {
using namespace std::meta;

#if defined(_GLIBCXX_META)
// A statement that condition_of accepts. condition_of looks through an
// expression statement, so `c ? f() : g();` is left out explicitly.
consteval bool is_if_statement(info s) {
  if (!std::meta::is_statement(s) || std::meta::is_expression(s) ||
      std::meta::is_expression_statement(s))
    return false;
  try {
    (void)std::meta::condition_of(s);
    return true;
  } catch (const std::meta::exception &) {
    return false;
  }
}
#endif
} // namespace ad::meta

#endif
