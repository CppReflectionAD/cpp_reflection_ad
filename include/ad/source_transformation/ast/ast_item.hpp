#pragma once

#include <string>

#include "../../autograd.h"
#include "../../meta.h"

namespace ad {
using std::meta::info;
namespace m = ad::meta;

enum class ASTKind {
  Expression,
  Statement
};

struct ASTItem;

template<typename T>
concept anyASTItem = std::is_base_of_v<ASTItem, T>;

struct ASTItem {
  ASTKind kind;

  ASTItem* parent = nullptr;

  constexpr ASTItem(ASTKind kind) : kind(kind) {}

  constexpr bool hasParent() {
    return nullptr != parent;
  }

  constexpr virtual ASTItem* clone() = 0;

  template<anyASTItem A, anyASTItem B>
  constexpr void setMember(A*& member, B* n) requires std::is_base_of_v<A, B> {
    member = n;
    member->parent = this;
  }

#define DECL_IS(name) \
  constexpr bool is ## name() { \
    return ASTKind:: name == kind; \
  }

  DECL_IS(Expression)
  DECL_IS(Statement)

#undef DECL_IS
};
}