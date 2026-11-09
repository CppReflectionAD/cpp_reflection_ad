#pragma once

#include "generation_context.hpp"

#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>

#include "../autograd.h"
#include "../meta.h"
#include "operator_info.hpp"
#include "string_util.hpp"
#include "throw_helper.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;
namespace su = ad::string_util;

struct ForwardExpression {
  std::string primal;
  std::string fwd;
};

struct ReverseExpression {
  std::string primal = {};

  std::vector<std::string> reverse_stmts = {};
  std::vector<std::string> lhs_values = {};
};

struct ExpressionAnalysis {
  static consteval std::ranges::range_value_t<std::string&> const* generatePrimal(info e, GenerationContext& ctx) {
    info exp = ad::detail::stripCasts(e); // TODO: Handle casts.

    std::string r = "";
    if (m::is_operator_expression(exp)) {
      auto operands = operands_of(exp);

      if (operands.size() != 2) {
        throw "Only binary handling for now"; // TODO: Implement
      }

      OperatorInfo op = getOperatorInfo(m::expression_operator_of(exp));
      r += su::join(
        ExpressionAnalysis::generatePrimal(operands[0], ctx),
        " ",
        op.symbol,
        " ",
        ExpressionAnalysis::generatePrimal(operands[1], ctx));
    } else if (m::is_literal(exp)) {
      r += m::display_string_of(constant_of(exp));
    } else if (m::is_id_expression(exp)) {
      r += identifier_of(declaration_of(exp));
    } else if (m::is_function_call(exp)) {
      if (ctx.reverse_generation) {
        std::string temporary = ctx.generateTemporary();

        ReverseExpression rws_exp = {};
        generateReverseFunctionCall(exp, temporary + "_b", rws_exp, ctx);

        for (std::string cur : rws_exp.lhs_values | std::views::reverse) {
          ctx.addReverseStatement(su::join(cur, " = 0.0;"));
        }
        for (std::string cur : rws_exp.reverse_stmts | std::views::reverse) {
          ctx.addReverseStatement(su::join(cur, ";"));
        }
        ctx.addStatement(su::join("auto ", temporary, " = ", rws_exp.primal, ";")); // TODO: Get proper type of return value.
        ctx.addStatement(su::join("auto ", temporary, "_b = 0.0;")); // TODO: Get proper type of return value.
        ctx.pushFunctionTemporary(temporary);

        r += temporary;
      } else {
        auto ops = m::operands_of(exp);
        std::string_view name = m::identifier_of(m::declaration_of(m::callee_of(exp)));

        std::string ops_s = "";
        for (info o : ops) {
          ops_s += su::join(ops_s.size() ? ", " : "", ExpressionAnalysis::generatePrimal(o, ctx));
        }

        r += su::join(name, "(", ops_s , ")");
      }
    } else {
      throwKind(exp);
    }

    return std::define_static_string(r);
  }

  // TODO: Hack to get function calls since meta::info objects can not be used in constexpr context.
  static consteval ForwardExpression getFunctionCalls(info e, GenerationContext& ctx) {
    info exp = ad::detail::stripCasts(e); // TODO: Handle casts.

    ForwardExpression r = {};
    if (m::is_operator_expression(exp)) {
      auto operands = operands_of(exp);

      for (info operand : operands) {
        getFunctionCalls(operand, ctx);
      }
    } else if (m::is_literal(exp)) {
      // Nothing
    } else if (m::is_id_expression(exp)) {
      // Nothing
    } else if (m::is_function_call(exp)) {
      // First handle arguments
      auto operands = operands_of(exp);
      for (info operand : operands) {
        getFunctionCalls(operand, ctx);
      }

      // Add function call to ctx
      info decl = m::declaration_of(m::callee_of(exp));
      ctx.addRequiredFunction(decl);
    } else {
      throwKind(exp);
    }

    return r;
  }

  static consteval std::vector<std::string> listOverwrites(info e, GenerationContext ctx) {
    info exp = ad::detail::stripCasts(e); // TODO: Handle casts.

    std::vector<std::string> list = {};
    if (m::is_operator_expression(exp)) {
      auto operands = operands_of(exp);

      if (operands.size() != 2) {
        throw "Only binary handling for now"; // TODO: Implement
      }

      OperatorInfo op = getOperatorInfo(m::expression_operator_of(exp));
      if (op.kind == OperatorKind::Assign) {
        list.push_back(ExpressionAnalysis::generatePrimal(operands[0], ctx));
      }
    } else if (m::is_literal(exp)) {
      // Nothing
    } else if (m::is_id_expression(exp)) {
      // Nothing
    } else if (m::is_function_call(exp)) {
      // Nothing
      // TODO: Add out and inout arguments of function.
    } else {
      throwKind(exp);
    }

    return list;
  }

  static consteval void generateReverseFunctionCall(info exp, std::string adj, ReverseExpression& data, GenerationContext& ctx) {
    auto ops = m::operands_of(exp);
    std::string_view name = m::identifier_of(m::declaration_of(m::callee_of(exp)));

    std::string ops_primal = "";
    std::string ops_rws = "";
    for (info o : ops) {
      std::string var_name = ExpressionAnalysis::generatePrimal(o, ctx); // TODO: Assume this is variable. Expressions would need to be handled in a special way.
      ops_primal += su::join(ops_primal.size() ? ", " : "", var_name);
      ops_rws += su::join(ops_rws.size() ? ", " : "", var_name, ", ", var_name, "_b");
    }
    ops_rws += su::join(ops_rws.size() ? ", " : "", adj);
    // ctx.addReverseStatement(su::join(adj, " = 0.0;")); // TODO: Handle reset of return properly.
    ctx.addReverseStatement(su::join(name, "_b(", ops_rws, ");"));

    data.primal =  su::join(name, "(", ops_primal , ")");

    ctx.addRequiredFunction(m::declaration_of(m::callee_of(exp)));
  }

  static consteval void generateReverse(info e, std::string adj, ReverseExpression& data, GenerationContext& ctx) {
    info exp = ad::detail::stripCasts(e); // TODO: Handle casts.

    if (m::is_operator_expression(exp)) {
      auto operands = operands_of(exp);

      if (operands.size() != 2) {
        throw "Only binary handling for now"; // TODO: Implement
      }

      OperatorInfo op = getOperatorInfo(m::expression_operator_of(exp));


      if (op.kind == OperatorKind::Assign) {
        if (!adj.empty()) {
          throw "Adjoint needs to be zero for assignment operators.";
        }

        std::string var_name = generatePrimal(operands[0], ctx);
        std::string adj_name = su::join(var_name, "_b"); // TODO: Remove shortcut that might fail  e.g. a[0] -> a[0]_b
        if (m::is_function_call(operands[1])) {
          generateReverseFunctionCall(operands[1], adj_name, data, ctx);
        } else {
          generateReverse(operands[1], adj_name, data, ctx);
          data.lhs_values.push_back(adj_name);
        }

        data.primal = su::join(var_name, " = ", data.primal, ";");
      } else if (op.kind == OperatorKind::Mul) {
        std::string p0 = generatePrimal(operands[0], ctx);
        std::string p1 = generatePrimal(operands[1], ctx);
        generateReverse(operands[1], su::join("(", p0, ") * ", adj), data, ctx);
        generateReverse(operands[0], su::join("(", p1, ") * ", adj), data, ctx);

        data.primal = su::join(p0, " * ", p1);
      } else {
        throw "Unkown operator";
      }
    } else if (m::is_literal(exp)) {
      // Nothing
      data.primal = generatePrimal(exp, ctx);
    } else if (m::is_id_expression(exp)) {
      data.reverse_stmts.push_back(su::join(m::identifier_of(declaration_of(exp)), "_b += ", adj));
      data.primal = generatePrimal(exp, ctx);
    } else if (m::is_function_call(exp)) {
      // Function calls have been handled in the primal generation. Just pop the temporary stack and add the update.
      std::string temporary = ctx.popFunctionTemporary();
      data.reverse_stmts.push_back(su::join(temporary, "_b += ", adj));
      data.primal = temporary;
      // data.reverse_stmts.push_back(su::join("func_todo", "_b += ", adj));
      // data.primal = "func_todo";
    } else {
      throwKind(exp);
    }
  }
};
}