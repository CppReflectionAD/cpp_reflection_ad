#pragma once

#include <string>

#include "../../autograd.h"
#include "../../meta.h"
#include "../throw_helper.hpp"
#include "ast_item.hpp"
#include "expressions.hpp"
#include "statements.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;



struct AstWalker {

  using StmtVector = std::vector<Statement*>;
  using StmtIter = typename StmtVector::iterator;

  struct StmtStackEntry {
    StmtVector* vec;
    StmtIter* iter;
    bool advance;
  };

  std::vector<StmtStackEntry> stmtInsertStack = {};

#define DECL_HANDLE(type, name) \
  constexpr virtual void handle(type* name) { \
    (void)name; \
  } \
  constexpr virtual void walk(type* name) { \
    handle(name); \
    name->walk(this); \
  }

  DECL_HANDLE(IdExpression, expr)
  DECL_HANDLE(FunctionCallExpression, expr)
  DECL_HANDLE(OperatorExpression, expr)
  DECL_HANDLE(LiteralExpression, expr)

  DECL_HANDLE(CommentStatement, stmt)
  DECL_HANDLE(DeclarationStatement, stmt)
  DECL_HANDLE(EmptyLineStatement, stmt)
  DECL_HANDLE(ExpressionStatement, stmt)
  DECL_HANDLE(ReturnStatement, stmt)

#undef DECL_HANDLE

  constexpr virtual void walk(Expression*& expr) {
    if (expr->isOperator()) {
      walk(OperatorExpression::cast(expr));
    } else if (expr->isLiteral()) {
      walk(LiteralExpression::cast(expr));
    } else if (expr->isFunctionCall()) {
      walk(FunctionCallExpression::cast(expr));
    } else if (expr->isId()) {
      walk(IdExpression::cast(expr));
    } else {
      throw "Unknown expression";
    }
  }

  constexpr virtual void walk(Statement*& stmt) {
#define CASE(name) \
  else if (stmt->is ## name()) { \
    walk(name ## Statement::cast(stmt)); \
  }

    if (false) {}
    CASE(Comment)
    CASE(Declaration)
    CASE(EmptyLine)
    CASE(Expression)
    CASE(Return)
    else {
      throw "Unknown statement.";
    }

#undef CASE
  }

  constexpr virtual void walk(std::vector<Statement*>* stmts) {
    StmtIter iter = stmts->begin();

    pushInsertStack(stmts, &iter);
    while(iter != stmts->end()) {
      Statement* stmt = *iter;

      walk(stmt);

      if (stmtInsertStack.back().advance) {
        ++iter;
      }
      stmtInsertStack.back().advance = true;
    }
    popInsertStack();
  }

  constexpr void pushInsertStack(StmtVector* vec, StmtIter* iter) {
    stmtInsertStack.push_back({vec, iter, true});
  }

  constexpr void popInsertStack() {
    if (stmtInsertStack.empty()) {
      throw "Can not pop from empty insert stack.";
    }
    stmtInsertStack.pop_back();
  }

  constexpr void removeStatement() {
    if (stmtInsertStack.empty()) {
      throw "No statement body for insertion.";
    }

    StmtStackEntry& b = stmtInsertStack.back();

    StmtIter new_iter = b.vec->erase(*b.iter);

    *b.iter = new_iter;
    b.advance = false; // Do not advance, since we already point to the next element.
  }

  constexpr void insertStatement(Statement* stmt) {
    if (stmtInsertStack.empty()) {
      throw "No statement body for insertion.";
    }

    StmtStackEntry& b = stmtInsertStack.back();

    StmtIter new_iter = b.vec->insert(*b.iter, stmt);

    // Advance again to the cur_element
    ++new_iter;
    *b.iter = new_iter;
  }
};

}