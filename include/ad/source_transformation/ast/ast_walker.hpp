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
  using StmtStackEntry = std::pair<StmtVector*, StmtIter*>;

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

  DECL_HANDLE(DeclarationStatement, stmt)
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

  constexpr virtual void walk(std::vector<Statement*>* stmts) {
    StmtIter iter = stmts->begin();

    pushInsertStack(stmts, &iter);
    for (; iter != stmts->end(); ++iter) {
      Statement* stmt = *iter;

      if (stmt->isDeclaration()) {
        walk(DeclarationStatement::cast(stmt));
      }
      else if (stmt->isExpression()) {
        walk(ExpressionStatement::cast(stmt));
      }
      else if (stmt->isReturn()) {
        walk(ReturnStatement::cast(stmt));
      }
      else {
        throw "Unknown expression.";
      }
    }
    popInsertStack();
  }

  constexpr void pushInsertStack(StmtVector* vec, StmtIter* iter) {
    stmtInsertStack.push_back({vec, iter});
  }

  constexpr void popInsertStack() {
    if (stmtInsertStack.empty()) {
      throw "Can not pop from empty insert stack.";
    }
    stmtInsertStack.pop_back();
  }

  constexpr void insertStatement(Statement* stmt) {
    if (stmtInsertStack.empty()) {
      throw "No statement body for insertion.";
    }

    StmtVector* vec = stmtInsertStack.back().first;
    StmtIter* iter = stmtInsertStack.back().second;

    StmtIter new_iter = vec->insert(*iter, stmt);

    // Advance again to the cur_element
    ++new_iter;
    *iter = new_iter;
  }
};

}