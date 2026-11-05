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

#include "meta_ast.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;
namespace su = ad::string_util;

struct ToCppWalker : public AstWalker {

  std::string code = {};

  using AstWalker::walk;

  constexpr void handle(OperatorExpression* expr) override {
    OperatorInfo op = getOperatorInfo(expr->op);
    code += su::joinE(op.symbol, " ");
  }

  constexpr void handle(IdExpression* expr) override {
    code += su::joinE(expr->id, " ");
  }

  constexpr void walk(DeclarationStatement* stmt) override {
    code += su::joinE(stmt->type_name, " ", stmt->var_name, " = ");
    walk(stmt->initializer);
    code += ";\n";
  }

  constexpr void walk(ExpressionStatement* stmt) override {
    walk(stmt->expression);
    code += ";\n";
  }

  constexpr void walk(ReturnStatement* stmt) override {
    code += "return ";
    walk(stmt->return_expression);
    code += ";\n";
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

    ctx.addIndent();
    for (info stmt : m::statements_of(m::body_of(func))) {
      r += ctx.format(StatementAnalysis::generatePrimal(stmt, ctx) + "\n");
    }
    ctx.removeIndent();
    r += "}\n";

    std::vector<Statement*> stmts = fromFunctionBody(m::body_of(func));
    ToCppWalker walker = {};
    walker.walk(&stmts);

    r += walker.code;

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

    ctx.addIndent();
    for (info stmt : m::statements_of(m::body_of(func))) {
      StatementAnalysis::generateForward(stmt, ctx);
    }
    r += ctx.flushStatements();
    ctx.removeIndent();
    r += "}\n";

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

