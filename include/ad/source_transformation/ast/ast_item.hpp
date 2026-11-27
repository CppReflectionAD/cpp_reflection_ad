#pragma once

#include <string>

#include "../../autograd.h"
#include "../../meta.h"

namespace ad {

struct NoType {};

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

  constexpr bool is(ASTKind k) const {
    return k == kind;
  }
};
}