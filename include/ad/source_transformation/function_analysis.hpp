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

  std::string code = {};
  GenerationContext* ctx;

  constexpr ExtractFunctionWalker(GenerationContext* ctx) : ctx(ctx) {}

  using AstWalker::walk;

  constexpr void walk(Expression*& e) override {
    if (e->isFunctionCall()) {
      FunctionCallExpression* expr = FunctionCallExpression::cast(e);

      // First walk children
      AstWalker::walk(expr);

      std::string temporary_name = ctx->generateTemporary();
      DeclarationStatement* temporary_decl = new DeclarationStatement("double", temporary_name, expr);
      AstWalker::insertStatement(temporary_decl);

      IdExpression* temporary_id = new IdExpression(temporary_name);

      // Replace the expression
      e = temporary_id;
    } else {
      AstWalker::walk(e);
    }
  }
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

    ToCppWalker walker = {&ctx};
    ctx.addIndent();

    walker.walk(&stmts);
    r += walker.code;

    ctx.removeIndent();
    // ctx.addIndent();
    // for (info stmt : m::statements_of(m::body_of(func))) {
    //   StatementAnalysis::generateForward(stmt, ctx);
    // }
    // r += ctx.flushStatements();
    // ctx.removeIndent();
    r += "}\n";

    releaseFunctionBody(stmts);

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

