#pragma once

#include "to_cpp_walker.hpp"

#include <ranges>
#include <sstream>

#include "../ast/ast_walker.hpp"
#include "../generation_context.hpp"
#include "../operator_info.hpp"
#include "../string_util.hpp"

namespace ad {
namespace su = ad::string_util;

constexpr std::string genReverseADName(std::string_view const& name) {
  return su::joinE(name, "_b");
}

struct GetOverwritesWalker : public AstWalker {
  std::vector<Expression*> overwrites = {};

  using AstWalker::walk;
  using AstWalker::handle;

  constexpr void handle(OperatorExpression* expr) override {
    OperatorInfo op = getOperatorInfo(expr->op);
    if (expr->operands.size() == 2 && OperatorKind::Assign == op.kind) {
      overwrites.push_back(expr->operands[0]);
    }
  }
};

struct ReverseADExpressionWalker : public AstWalker {
  std::vector<Expression*> updates = {};
  std::vector<Expression*> resets = {};

  using AstWalker::walk;
  using AstWalker::handle;

  constexpr void walk(OperatorExpression* expr) override {
    std::vector<Expression*> op_d = handleOperators(expr->operands);

    OperatorInfo op = getOperatorInfo(expr->op);
    if (op_d.size() == 1) {
      throw "Handle unary operators";
    } else if (op_d.size() == 2) {
      if (OperatorKind::Add == op.kind) {
        ad_expr = (wrap(op_d[0]) + wrap(op_d[1])).wrap;
      }
      else if (OperatorKind::Assign == op.kind) {
        ad_expr = (wrap(op_d[0]) = wrap(op_d[1])).wrap;
      }
      else if (OperatorKind::Mul == op.kind) {
        ad_expr = (wrap(expr->operands[1]->clone()) * wrap(op_d[0]) + wrap(expr->operands[0]->clone()) * wrap(op_d[1])).wrap;
      }
    } else {
      throw "Unhandled number of operands";
    }
  }

  constexpr void walk(IdExpression* expr) override {
    ad_expr = new IdExpression(genReverseADName(expr->id));
  }

  constexpr void walk(LiteralExpression* expr) override {
    (void) expr;

    ad_expr = new LiteralExpression("0.0");
  }

  constexpr void walk(FunctionCallExpression* expr) override {

    std::vector<Expression*> ad_ops = handleOperators(expr->operands);

    std::vector<Expression*> all_ops(ad_ops.size() * 2);
    for (size_t i = 0; i < ad_ops.size(); i += 1) {
      all_ops[i * 2] = expr->operands[i]->clone();
      all_ops[i * 2 + 1] = ad_ops[i];
    }

    ad_expr = new FunctionCallExpression(genReverseADName(expr->func_name), all_ops);
  }

private:
  constexpr std::vector<Expression*> handleOperators(std::vector<Expression*>& ops) {
    std::vector<Expression*> operands(ops.size());

    for (size_t i = 0; i < operands.size(); i += 1) {
      ReverseADExpressionWalker w = {};
      w.walk(ops[i]);
      operands[i] = w.ad_expr;
    }

    return operands;
  }
};



struct ReverseADWalker : public AstWalker {
  GenerationContext* ctx;

  std::vector<Statement*> reverse_statements = {};

  constexpr ReverseADWalker(GenerationContext* ctx) : ctx(ctx) {}

  using AstWalker::walk;

  constexpr void walk(DeclarationStatement* stmt) override {
    reverse_statements.push_back(new DeclarationStatement(stmt->type_name, genReverseADName(stmt->var_name), new LiteralExpression("0.0")));

    // ReverseADExpressionWalker init_ad_expr = {};
    // init_ad_expr.walk(stmt->initializer);
    //
    // if (init_ad_expr.ad_expr->isFunctionCall()) {
    //   freeExpression(stmt->initializer);
    //
    //   stmt->type_name = "auto";
    //   stmt->var_name = genFunctionVar(stmt->var_name);
    //   stmt->setMember(stmt->initializer, init_ad_expr.ad_expr);
    // }
    // else {
    //   AstWalker::insertStatement(new DeclarationStatement(stmt->type_name, genReverseADName(stmt->var_name), init_ad_expr.ad_expr));
    // }
  }

  constexpr void walk(ExpressionStatement* stmt) override {
    GetOverwritesWalker overwrites = {};
    overwrites.walk(stmt->expression);

    if (!overwrites.overwrites.empty()) {
      std::vector<Expression*> clones_store = {};
      std::vector<Expression*> clones_restore = {};
      for (Expression* cur : overwrites.overwrites) {
        clones_store.push_back(cur->clone());
        clones_restore.push_back(cur->clone());
      }

      insertStatement(new ExpressionStatement(new FunctionCallExpression("store", clones_store)));
      reverse_statements.push_back(new ExpressionStatement(new FunctionCallExpression("restore", clones_restore)));
    }
    // bool function_handled = false;
    // if (stmt->expression->isOperator()) {
    //   OperatorExpression* op_expr = OperatorExpression::cast(stmt->expression);
    //   OperatorInfo op_info = getOperatorInfo(op_expr->op);
    //
    //   if (OperatorKind::Assign == op_info.kind && op_expr->operands[1]->isFunctionCall()) {
    //     ReverseADExpressionWalker expr_ad_expr = {};
    //     expr_ad_expr.walk(op_expr->operands[1]);
    //
    //     // TODO: This assumes id assignment. Improve when structured bindings are supported.
    //     ExpressionToCppWalker lhsName = {};
    //     lhsName.walk(op_expr->operands[0]);
    //
    //     freeExpression(op_expr->operands[0]);
    //     freeExpression(op_expr->operands[1]);
    //
    //     op_expr->setMember(op_expr->operands[0], new IdExpression(genFunctionVar(lhsName.code)));
    //     op_expr->setMember(op_expr->operands[1], expr_ad_expr.ad_expr);
    //
    //     function_handled = true;
    //   }
    // }
    //
    // if (!function_handled) {
    //   ReverseADExpressionWalker expr_ad_expr = {};
    //   expr_ad_expr.walk(stmt->expression);
    //
    //   AstWalker::insertStatement(new ExpressionStatement(expr_ad_expr.ad_expr));
    // }
  }

  constexpr void walk(ReturnStatement* stmt) override {
    (void)stmt;
    // ReverseADExpressionWalker expr_ad_expr = {};
    // expr_ad_expr.walk(stmt->return_expression);
    //
    // if (expr_ad_expr.ad_expr->isFunctionCall()) {
    //   freeExpression(stmt->return_expression);
    //   stmt->setMember(stmt->return_expression, expr_ad_expr.ad_expr);
    // }
    // else {
    //   Expression* primal_expr = stmt->return_expression;
    //   stmt->setMember(stmt->return_expression, new FunctionCallExpression("std::make_pair", {primal_expr, expr_ad_expr.ad_expr}));
    // }
  }

private:
  constexpr std::string genFunctionVar(std::string_view const& var) {
    return su::joinE("[", var, ", ", genReverseADName(var), "]");
  };
};
}

