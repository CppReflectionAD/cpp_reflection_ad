#pragma once

#include <sstream>
#include <fstream>
#include <ranges>
#include <iostream>

#include "../autograd.h"
#include "../meta.h"
#include "generation_context.hpp"
#include "operator_info.hpp"
#include "string_util.hpp"
#include "throw_helper.hpp"

#include "expression_analysis.hpp"
#include "function_analysis.hpp"

#include "ast/meta_ast.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;
namespace su = ad::string_util;

struct ExpressionToCppWalker : public AstWalker {
  std::string code = {};

  using AstWalker::walk;
  using AstWalker::handle;

  constexpr void walk(OperatorExpression* expr) override {
    std::vector<std::string> operands = handleOperators(expr->operands);

    OperatorInfo op = getOperatorInfo(expr->op);
    if (operands.size() == 1) {
      // TODO: Prefix and postfix distinction.
      code += su::joinE(op.symbol, " ", operands[0]);
    } else if (operands.size() == 2) {
      code += su::joinE(operands[0], " ", op.symbol, " ", operands[1]);
    } else {
      throw "Unhandled number of operands";
    }
  }

  constexpr void walk(IdExpression* expr) override {
    code += expr->id;
  }

  constexpr void walk(LiteralExpression* expr) override {
    code += expr->literal;
  }

  constexpr void walk(FunctionCallExpression* expr) override {
    std::vector<std::string> operands = handleOperators(expr->operands);

    code += su::joinE(expr->func_name, "(");
    bool first = true;
    for (std::string const& op : operands) {
      if (!first) {
        code += ", ";
      }
      code += op;

      first = false;
    }
    code += ")";
  }

private:
  constexpr std::vector<std::string> handleOperators(std::vector<Expression*>& ops) {
    std::vector<std::string> operands(ops.size());

    for (size_t i = 0; i < operands.size(); i += 1) {
      ExpressionToCppWalker w = {};
      w.walk(ops[i]);
      operands[i] = w.code;
    }

    return operands;
  }
};

struct ToCppWalker : public AstWalker {

  std::string code = {};
  GenerationContext* ctx;

  constexpr ToCppWalker(GenerationContext* ctx) : ctx(ctx) {}

  using AstWalker::walk;

  constexpr void walk(DeclarationStatement* stmt) override {
    ExpressionToCppWalker init_str = {};
    init_str.walk(stmt->initializer);

    code += ctx->format(su::joinE(stmt->type_name, " ", stmt->var_name, " = ", init_str.code, ";\n"));
  }

  constexpr void walk(ExpressionStatement* stmt) override {
    ExpressionToCppWalker expr_str = {};
    expr_str.walk(stmt->expression);

    code += ctx->format(su::joinE(expr_str.code, ";\n"));
  }

  constexpr void walk(ReturnStatement* stmt) override {
    ExpressionToCppWalker expr_str = {};
    expr_str.walk(stmt->return_expression);

    code += ctx->format(su::joinE("return ", expr_str.code, ";\n"));
  }
};

struct ExtractFunctionWalker : public AstWalker {

  GenerationContext* ctx;

  constexpr ExtractFunctionWalker(GenerationContext* ctx) : ctx(ctx) {}

  using AstWalker::walk;

  constexpr void walk(Expression*& e) override {
    bool handle = e->isFunctionCall();
    if (handle && e->hasParent() && e->parent->isStatement()) {
      Statement* parent = Statement::cast(e->parent);

      // Do not extract direct assignments
      handle = !parent->isDeclaration();
    }

    if (handle && e->hasParent() && e->parent->isExpression()) {
      Expression* parent = Expression::cast(e->parent);

      if (parent->isOperator()) {
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

constexpr std::string genForwardADName(std::string_view const& name) {
  return su::joinE(name, "_d");
}

struct ForwardADExpressionWalker : public AstWalker {
  Expression* ad_expr = nullptr;


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
    ad_expr = new IdExpression(genForwardADName(expr->id));
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

    ad_expr = new FunctionCallExpression(genForwardADName(expr->func_name), all_ops);
  }

private:
  constexpr std::vector<Expression*> handleOperators(std::vector<Expression*>& ops) {
    std::vector<Expression*> operands(ops.size());

    for (size_t i = 0; i < operands.size(); i += 1) {
      ForwardADExpressionWalker w = {};
      w.walk(ops[i]);
      operands[i] = w.ad_expr;
    }

    return operands;
  }
};

struct ForwardADWalker : public AstWalker {
  GenerationContext* ctx;

  constexpr ForwardADWalker(GenerationContext* ctx) : ctx(ctx) {}

  using AstWalker::walk;

  constexpr void walk(DeclarationStatement* stmt) override {
    ForwardADExpressionWalker init_ad_expr = {};
    init_ad_expr.walk(stmt->initializer);

    if (init_ad_expr.ad_expr->isFunctionCall()) {
      freeExpression(stmt->initializer);

      stmt->type_name = "auto";
      stmt->var_name = genFunctionVar(stmt->var_name);
      stmt->setMember(stmt->initializer, init_ad_expr.ad_expr);
    }
    else {
      AstWalker::insertStatement(new DeclarationStatement(stmt->type_name, genForwardADName(stmt->var_name), init_ad_expr.ad_expr));
    }
  }

  constexpr void walk(ExpressionStatement* stmt) override {

    bool function_handled = false;
    if (stmt->expression->isOperator()) {
      OperatorExpression* op_expr = OperatorExpression::cast(stmt->expression);
      OperatorInfo op_info = getOperatorInfo(op_expr->op);

      if (OperatorKind::Assign == op_info.kind && op_expr->operands[1]->isFunctionCall()) {
        ForwardADExpressionWalker expr_ad_expr = {};
        expr_ad_expr.walk(op_expr->operands[1]);

        // TODO: This assumes id assignment. Improve when structured bindings are supported.
        ExpressionToCppWalker lhsName = {};
        lhsName.walk(op_expr->operands[0]);

        freeExpression(op_expr->operands[0]);
        freeExpression(op_expr->operands[1]);

        op_expr->setMember(op_expr->operands[0], new IdExpression(genFunctionVar(lhsName.code)));
        op_expr->setMember(op_expr->operands[1], expr_ad_expr.ad_expr);

        function_handled = true;
      }
    }

    if (!function_handled) {
      ForwardADExpressionWalker expr_ad_expr = {};
      expr_ad_expr.walk(stmt->expression);

      AstWalker::insertStatement(new ExpressionStatement(expr_ad_expr.ad_expr));
    }
  }

  constexpr void walk(ReturnStatement* stmt) override {
    ForwardADExpressionWalker expr_ad_expr = {};
    expr_ad_expr.walk(stmt->return_expression);

    if (expr_ad_expr.ad_expr->isFunctionCall()) {
      freeExpression(stmt->return_expression);
      stmt->setMember(stmt->return_expression, expr_ad_expr.ad_expr);
    }
    else {
      Expression* primal_expr = stmt->return_expression;
      stmt->setMember(stmt->return_expression, new FunctionCallExpression("std::make_pair", {primal_expr, expr_ad_expr.ad_expr}));
    }
  }

private:
  constexpr std::string genFunctionVar(std::string_view const& var) {
    return su::joinE("[", var, ", ", genForwardADName(var), "]");
  };
};


struct FunctionAnalysis {
  static consteval auto generatePrimal(info func) {
    GenerationContext ctx = {};
    std::string r = "";
    info ret_type = m::return_type_of(func);
    auto params = m::parameters_of(func);

    std::string params_s = "";
    for (info p : params) {
      params_s += su::join(params_s.size() ? ", " : "", m::display_string_of(m::type_of(p)), " ", m::identifier_of(p));
    }

    r += su::join(m::display_string_of(ret_type), " ", m::identifier_of(func), "(", params_s, ") {\n");

    std::vector<Statement*> stmts = fromFunctionBody(m::body_of(func));
    ToCppWalker walker = {&ctx};
    ctx.addIndent();

    walker.walk(&stmts);
    r += walker.code;

    ctx.removeIndent();
    r += "}\n";

    releaseFunctionBody(stmts);

    return std::define_static_string(r);
  }

  static consteval auto generateForward(info func, GenerationContext& ctx) {
    std::string r = "";
    info ret_type = m::return_type_of(func);
    auto params = m::parameters_of(func);

    std::string params_s = "";
    for (info p : params) {
      params_s += su::join(
        params_s.size() ? ", " : "",
        m::display_string_of(m::type_of(p)), " ", m::identifier_of(p), ", ",
        m::display_string_of(m::type_of(p)), " ", m::identifier_of(p), "_d");
    }

    r += su::join("std::pair<", m::display_string_of(ret_type), ", ", m::display_string_of(ret_type), "> ", m::identifier_of(func), "_d(", params_s, ") {\n");

    std::vector<Statement*> stmts = fromFunctionBody(m::body_of(func));

    // Replace function calls with temporaries.
    ExtractFunctionWalker extFnc = {&ctx};
    extFnc.walk(&stmts);

    ForwardADWalker ad = {&ctx};
    ad.walk(&stmts);

    ToCppWalker walker = {&ctx};
    ctx.addIndent();

    walker.walk(&stmts);
    r += walker.code;

    ctx.removeIndent();
    r += "}\n";

    releaseFunctionBody(stmts);

    // TODO: Remove when function declarations are part of the AST.
    for (info stmt : m::statements_of(m::body_of(func))) {
      StatementAnalysis::getFunctionCalls(stmt, ctx);
    }

    return std::define_static_string(r);
  }

  static consteval auto generateReverseJoined(info func, GenerationContext& ctx) {
    std::string r = "";
    info ret_type = m::return_type_of(func);
    auto params = m::parameters_of(func);

    std::string params_s = "";
    for (info p : params) {
      params_s += su::join(
        params_s.size() ? ", " : "",
        m::display_string_of(m::type_of(p)), " ", m::identifier_of(p), ", ",
        m::display_string_of(m::type_of(p)), "& ", m::identifier_of(p), "_b");
    }
    if (!m::is_void_type(ret_type)) {
      params_s += su::join(
        params_s.size() ? ", " : "",
        m::display_string_of(ret_type), " ret_b");
    }

    // Reverse functions do not return values.
    r += su::join("void ", m::identifier_of(func), "_b(", params_s, ") {\n");

    ctx.addIndent();
    for (info stmt : m::statements_of(m::body_of(func))) {
      bool is_return = m::is_return_statement(stmt);

      StatementAnalysis::generateStore(stmt, ctx);
      StatementAnalysis::generateReverseDecl(stmt, ctx);
      StatementAnalysis::generateReverse(stmt, ctx, !is_return);
      StatementAnalysis::generateRestore(stmt, ctx);
      ctx.addReverseStatement(su::join("//", StatementAnalysis::generatePrimal(stmt, ctx)));
      ctx.addReverseStatement("");
    }

    r += ctx.flushStatements();
    r += ctx.flushReverseStatements();

    ctx.removeIndent();
    r += "}\n";

    return std::define_static_string(r);
  }
};
}

