#pragma once

#include <ranges>
#include <iostream>

#include "../autograd.h"
#include "../meta.h"
#include "generation_context.hpp"
#include "string_util.hpp"
#include "throw_helper.hpp"

#include "expression_analysis.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;
namespace su = ad::string_util;

struct StatementAnalysis {
  static consteval std::string generatePrimal(info stmt, GenerationContext& ctx) {
    (void)ctx;
    std::string r = "";
    if (m::is_declaration_statement(stmt)) {
      info v = declared_variable_of(stmt);
      r += su::join(
        m::display_string_of(m::type_of(v)),
        " ",
        m::identifier_of(v),
        " = ",
        ExpressionAnalysis::generatePrimal(initializer_of(v), ctx));
    } else if (m::is_expression_statement(stmt)) {
      r += ExpressionAnalysis::generatePrimal(expression_of(stmt), ctx);
    } else if (m::is_return_statement(stmt)) {
      r += su::join("return ", ExpressionAnalysis::generatePrimal(return_operand_of(stmt), ctx));
    } else {
      throwKind(stmt);
    }

    r += ";";
    return r;
  }

  // TODO: Hack to get function calls since meta::info objects can not be used in constexpr context.
  static consteval void getFunctionCalls(info stmt, GenerationContext& ctx) {
    if (m::is_declaration_statement(stmt)) {
      info v = declared_variable_of(stmt);
      ForwardExpression rhs = ExpressionAnalysis::getFunctionCalls(initializer_of(v), ctx);
    } else if (m::is_expression_statement(stmt)) {
      ForwardExpression rhs = ExpressionAnalysis::getFunctionCalls(expression_of(stmt), ctx);
    } else if (m::is_return_statement(stmt)) {
      ForwardExpression rhs = ExpressionAnalysis::getFunctionCalls(return_operand_of(stmt), ctx);
    } else {
      throwKind(stmt);
    }
  }

  static consteval void generateForward(info stmt, GenerationContext& ctx) {
    if (m::is_declaration_statement(stmt)) {
      info v = declared_variable_of(stmt);
      std::string_view type_name = m::display_string_of(m::type_of(v));
      std::string_view var_name = m::identifier_of(v);

      if (m::is_function_call(initializer_of(v))) {
        ForwardExpression rhs = ExpressionAnalysis::generateForwardFunctionCall(initializer_of(v), ctx);
        ctx.addStatement(su::join("auto [", var_name, ", ", var_name, "_d] = ", rhs.fwd, ";"));
      } else {
        ForwardExpression rhs = ExpressionAnalysis::generateForward(initializer_of(v), ctx);
        ctx.addStatement(su::join(
          type_name,
          " ",
          var_name, "_d",
          " = ",
          rhs.fwd, ";"));
        ctx.addStatement(su::join(
          type_name,
          " ",
          var_name,
          " = ",
          rhs.primal, ";"));
      }
    } else if (m::is_expression_statement(stmt)) {
      info exp = expression_of(stmt);
      bool is_func_call = m::is_operator_expression(exp) &&
        operands_of(exp).size() == 2 &&
          getOperatorInfo(m::expression_operator_of(exp)).kind == OperatorKind::Assign &&
            m::is_function_call(operands_of(exp)[1]);

      ForwardExpression rhs = ExpressionAnalysis::generateForward(exp, ctx);
      ctx.addStatement(rhs.fwd + ";");

      // Function calls are optimized, so we can omit the primal statement.
      if (!is_func_call) {
        ctx.addStatement(rhs.primal + ";");
      }
    } else if (m::is_return_statement(stmt)) {
      ForwardExpression ret = ExpressionAnalysis::generateForward(return_operand_of(stmt), ctx);

      ctx.addStatement(su::join(
        "return std::make_pair(",
        ret.primal,
        ", ",
        ret.fwd,
        ");"));
    } else {
      throwKind(stmt);
    }
  }

  static consteval void generateStore(info stmt, GenerationContext& ctx) {
    if (m::is_declaration_statement(stmt)) {
      // Nothing
    } else if (m::is_expression_statement(stmt)) {
      std::vector<std::string> overwrites = ExpressionAnalysis::listOverwrites(expression_of(stmt), ctx);
      for (std::string cur : overwrites) {
        ctx.addStatement(su::join("store(", cur, ");"));
      }
    } else if (m::is_return_statement(stmt)) {
      // Nothing
    } else {
      throwKind(stmt);
    }
  }

  static consteval void generateRestore(info stmt, GenerationContext& ctx) {
    if (m::is_declaration_statement(stmt)) {
      // Nothing
    } else if (m::is_expression_statement(stmt)) {
      std::vector<std::string> overwrites = ExpressionAnalysis::listOverwrites(expression_of(stmt), ctx);
      for (std::string cur : overwrites) {
        ctx.addReverseStatement(su::join("restore(", cur, ");"));
      }
    } else if (m::is_return_statement(stmt)) {
      // Nothing
    } else {
      throwKind(stmt);
    }
  }

  static consteval void generateReverse(info stmt, GenerationContext& ctx, bool generate_primal) {
    ctx.reverse_generation = true;

    ReverseExpression rws_exp = {};
    if (m::is_declaration_statement(stmt)) {
      info v = declared_variable_of(stmt);
      std::string adj = su::join(m::identifier_of(v), "_b");

      if (m::is_function_call(initializer_of(v))) {
        ExpressionAnalysis::generateReverseFunctionCall(m::initializer_of(v), adj, rws_exp, ctx);
      } else {
        ExpressionAnalysis::generateReverse(initializer_of(v), adj, rws_exp, ctx);
        rws_exp.lhs_values.push_back(adj);
      }
      rws_exp.primal = su::join(m::display_string_of(m::type_of(v)), " ", m::identifier_of(v), " = ", rws_exp.primal, ";");
    } else if (m::is_expression_statement(stmt)) {
      ExpressionAnalysis::generateReverse(expression_of(stmt), "", rws_exp, ctx);
    } else if (m::is_return_statement(stmt)) {
      std::string adj = "ret_b";
      ExpressionAnalysis::generateReverse(return_operand_of(stmt), adj, rws_exp, ctx);
      rws_exp.lhs_values.push_back(adj);

      rws_exp.primal = su::join("return ", rws_exp.primal);
    } else {
      throwKind(stmt);
    }

    ctx.reverse_generation = false;

    if (generate_primal) {
      ctx.addStatement(rws_exp.primal);
    }

    for (std::string cur : rws_exp.lhs_values | std::views::reverse) {
      ctx.addReverseStatement(su::join(cur, " = 0.0;"));
    }
    for (std::string cur : rws_exp.reverse_stmts | std::views::reverse) {
      ctx.addReverseStatement(su::join(cur, ";"));
    }
  }

  static consteval void generateReverseDecl(info stmt, GenerationContext& ctx) {
    if (m::is_declaration_statement(stmt)) {
      info v = declared_variable_of(stmt);
      ctx.addStatement(su::join(
        m::display_string_of(m::type_of(v)),
        " ",
        m::identifier_of(v),
        "_b = 0.0;"));
    } else if (m::is_expression_statement(stmt)) {
      // Nothing
    } else if (m::is_return_statement(stmt)) {
      // Nothing
    } else {
      throwKind(stmt);
    }
  }
};

}