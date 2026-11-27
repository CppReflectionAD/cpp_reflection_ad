#pragma once

#include <sstream>
#include <ranges>

#include "../ast/ast_walker.hpp"
#include "../generation_context.hpp"
#include "../operator_info.hpp"
#include "../string_util.hpp"

namespace ad {
namespace su = ad::string_util;

struct ExtractFunctionWalker : public AstWalker {
  GenerationContext* ctx;

  constexpr ExtractFunctionWalker(GenerationContext* ctx) : ctx(ctx) {}

  using AstWalker::walk;

  constexpr void walk(Expression*& e) override {
    bool handle = e->is(ExpressionKind::FunctionCall);
    if (handle && e->hasParent() && e->parent->is(ASTKind::Statement)) {
      Statement* parent = Statement::cast(e->parent);

      // Do not extract direct assignments
      handle = !parent->is(StatementKind::Declaration);
    }

    if (handle && e->hasParent() && e->parent->is(ASTKind::Expression)) {
      Expression* parent = Expression::cast(e->parent);

      if (parent->is(ExpressionKind::Operator)) {
        OperatorExpression* op_parent = OperatorExpression::cast(parent);
        OperatorInfo op = getOperatorInfo(op_parent->op);

        // Do not handle direct assignment.
        handle = !(OperatorKind::Assign == op.kind);
      }
    }

    if (handle) {
      FunctionCallExpression* expr = FunctionCallExpression::cast(e);
      ASTItem* parent = expr->parent;

      // First walk children
      AstWalker::walk(expr);

      std::string temporary_name = ctx->generateTemporary();
      DeclarationStatement* temporary_decl = new DeclarationStatement("double", temporary_name, expr);
      AstWalker::insertStatement(temporary_decl);

      parent->setMember(e, new IdExpression(temporary_name));
    } else {
      AstWalker::walk(e);
    }
  }
};
}

