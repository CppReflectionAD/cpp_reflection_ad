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
    if (expr->operands.size() == 2 && (OperatorKind::Assign == op.kind || OperatorKind::AddAssign == op.kind)) {
      overwrites.push_back(expr->operands[0]);
    }
  }
};

struct GetReverseAccessWalker : public AstWalker {
  Expression* reverse_access = nullptr;

  using AstWalker::walk;
  using AstWalker::handle;

  constexpr void walk(OperatorExpression* expr) override {
    (void)expr;
    throw "No operators for access supported.";
  }

  constexpr void walk(IdExpression* expr) override {
    reverse_access = new IdExpression(genReverseADName(expr->id));
  }

  constexpr void walk(LiteralExpression* expr) override {
    (void) expr;
    throw "No Literals for access supported.";
  }

  constexpr void walk(FunctionCallExpression* expr) override {
    (void) expr;
    throw "No function calls for access supported.";
  }
};

struct ReverseADExpressionContext {
  std::vector<Expression*> updates = {};
  std::vector<Expression*> resets = {};
  std::vector<Statement*> restores = {};
};

struct ReverseADExpressionWalker : public AstWalker {
  Expression* adj;

  ReverseADExpressionContext* rws_ctx;

  constexpr ReverseADExpressionWalker(Expression* adj, ReverseADExpressionContext* rws_ctx) : adj(adj), rws_ctx(rws_ctx) {}

  constexpr ~ReverseADExpressionWalker() {
    if (nullptr != adj) {
      freeExpression(adj);
    }
  }

  using AstWalker::walk;
  using AstWalker::handle;

  constexpr void walk(OperatorExpression* expr) override {
    OperatorInfo op = getOperatorInfo(expr->op);
    if (expr->operands.size() == 1) {
      throw "Handle unary operators";
    } else if (expr->operands.size() == 2) {
      if (OperatorKind::Add == op.kind) {
        // Factor of 1 just continue walk.
        walk(expr->operands[0]);
        walk(expr->operands[1]);
      }
      else if (OperatorKind::Assign == op.kind || OperatorKind::AddAssign == op.kind) {
        GetReverseAccessWalker reverse_access = {};
        reverse_access.walk(expr->operands[0]);

        if (OperatorKind::Assign == op.kind) {
          rws_ctx->resets.push_back(reverse_access.reverse_access->clone());
        }
        ReverseADExpressionWalker op1_walker = {reverse_access.reverse_access, rws_ctx};
        op1_walker.walk(expr->operands[1]);
      }
      else if (OperatorKind::Mul == op.kind) {
        ReverseADExpressionWalker op0_walker = {(wrap(expr->operands[1]->clone()) * wrap(adj->clone())).wrap, rws_ctx};
        op0_walker.walk(expr->operands[0]);

        ReverseADExpressionWalker op1_walker = {(wrap(expr->operands[0]->clone()) * wrap(adj->clone())).wrap, rws_ctx};
        op1_walker.walk(expr->operands[1]);
      }
    } else {
      throw "Unhandled number of operands";
    }
  }

  constexpr void walk(IdExpression* expr) override {
    rws_ctx->updates.push_back((wrap(new IdExpression(genReverseADName(expr->id))) += wrap(adj->clone())).wrap);
  }

  constexpr void walk(LiteralExpression* expr) override {
    (void) expr;
    // Do Nothing
  }

  constexpr void walk(FunctionCallExpression* expr) override {
    std::vector<Expression*> adj_operands = handleOperators(expr->operands);

    std::vector<Expression*> combined_operands(adj_operands.size() * 2 + 1);
    for (size_t i = 0; i < adj_operands.size(); i += 1) {
      combined_operands[i * 2] = expr->operands[i]->clone();
      combined_operands[i * 2 + 1] = adj_operands[i];
    }

    combined_operands.back() = adj->clone();
    rws_ctx->updates.push_back(new FunctionCallExpression(genReverseADName(expr->func_name), combined_operands));
  }

private:
  constexpr std::vector<Expression*> handleOperators(std::vector<Expression*>& ops) {
    std::vector<Expression*> operands(ops.size());

    for (size_t i = 0; i < operands.size(); i += 1) {
      GetReverseAccessWalker w = {};
      w.walk(ops[i]);
      operands[i] = w.reverse_access;
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
    // Add creation of adjoint
    insertStatement(new DeclarationStatement(stmt->type_name, genReverseADName(stmt->var_name), new LiteralExpression("0.0")));

    ReverseADExpressionContext rws_ctx = {};

    IdExpression* adj = new IdExpression(genReverseADName(stmt->var_name));
    ReverseADExpressionWalker rws = {adj->clone(), &rws_ctx};
    rws.walk(stmt->initializer);
    rws_ctx.resets.push_back(adj);

    addReverseContext(stmt, rws_ctx);
  }

  constexpr void walk(ExpressionStatement* stmt) override {
    // Add overwrites stores and restores
    GetOverwritesWalker overwrites = {};
    overwrites.walk(stmt->expression);

    ReverseADExpressionContext rws_ctx = {};

    if (!overwrites.overwrites.empty()) {
      std::vector<Expression*> clones_store = {};
      std::vector<Expression*> clones_restore = {};
      for (Expression* cur : overwrites.overwrites) {
        clones_store.push_back(cur->clone());
        clones_restore.push_back(cur->clone());
      }

      insertStatement(new ExpressionStatement(new FunctionCallExpression("store", clones_store)));
      rws_ctx.restores.push_back(new ExpressionStatement(new FunctionCallExpression("restore", clones_restore)));
    }

    ReverseADExpressionWalker rws = {nullptr, &rws_ctx};
    rws.walk(stmt->expression);

    addReverseContext(stmt, rws_ctx);
  }

  constexpr void walk(ReturnStatement* stmt) override {
    ReverseADExpressionContext rws_ctx = {};
    IdExpression* adj = new IdExpression(genReverseADName("ret"));

    ReverseADExpressionWalker rws = {adj->clone(), &rws_ctx};
    rws.walk(stmt->return_expression);

    rws_ctx.resets.push_back(adj);

    addReverseContext(stmt, rws_ctx);

    Statement* release = stmt;
    releaseStatement(release);
    removeStatement();
  }

private:

  constexpr void addReverseContext(Statement* stmt, ReverseADExpressionContext& rws_ctx) {
    for (Expression* cur : rws_ctx.resets) {
      reverse_statements.push_back(new ExpressionStatement((wrap(cur) = wrap(new LiteralExpression("0.0"))).wrap));
    }

    for (Expression* cur : rws_ctx.updates) {
      reverse_statements.push_back(new ExpressionStatement(cur));
    }

    for (Statement* cur : rws_ctx.restores) {
      reverse_statements.push_back(cur);
    }

    ToCppWalker to_cpp = {ctx};
    to_cpp.walk(stmt);
    reverse_statements.push_back(new CommentStatement(su::joinE("// ", to_cpp.code)));
    reverse_statements.push_back(new EmptyLineStatement());
  }

  constexpr std::string genFunctionVar(std::string_view const& var) {
    return su::joinE("[", var, ", ", genReverseADName(var), "]");
  };
};
}

