// Known clang gap: a member access `p.x` doesn't give its member.
//
// P4395 §3.5 leaves member access (`.`, `->`) to "dedicated predicates in a
// future revision". The clang fork has the expression (its private kind 5,
// MemberExpr), and operands_of() gives the object `p`, but nothing gives the
// member `x`: declaration_of() and callee_of() only accept a plain name
// (DeclRefExpr), and no metafunction says whether it was `.` or `->`. So
// ihm_expressions.hpp prints member accesses and member function calls such
// as `v.size()` as `<expr>`. The AoS-to-SoA use case (P4395 §2.5) needs the
// member too.
//
// The fix in the fork is in clang/lib/AST/ExprConstantMeta.cpp: let
// expression_declaration_of accept a MemberExpr and return its
// getMemberDecl(), let expression_callee_of return the method of a
// CXXMemberCallExpr, and expose MemberExpr::isArrow().
//
// This file expects the error recorded in member_access_declaration.fail.txt.
// When the fork fixes the gap, move this file to tests/clang_only/, delete the
// .fail.txt, and print member accesses in ihm_expressions.hpp.
#include <meta>

namespace m = std::meta;

struct point {
  double x;
  double y;
};

constexpr point p{1.0, 2.0};
constexpr double px = p.x;

// The initializer of `var`, without its conversions.
consteval m::info initializer(m::info var) {
  m::info e = m::initializer_of(var);
  while (m::is_cast(e))
    e = m::operands_of(e)[0];
  return e;
}

// What the fork gives: the object of `p.x`, as its only operand.
static_assert(m::operands_of(initializer(^^px)).size() == 1);
static_assert(m::declaration_of(m::operands_of(initializer(^^px))[0]) == ^^p);

// What it doesn't: the member.
static_assert(m::declaration_of(initializer(^^px)) == ^^point::x);

int main() {}
