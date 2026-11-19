#pragma once

#include <sstream>
#include <ranges>

#include "../ast/ast_walker.hpp"
#include "../generation_context.hpp"
#include "../operator_info.hpp"
#include "../string_util.hpp"

namespace ad {
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

  constexpr void walk(CommentStatement* stmt) override {
    code += ctx->format(stmt->comment);
  }

  constexpr void walk(DeclarationStatement* stmt) override {
    ExpressionToCppWalker init_str = {};
    init_str.walk(stmt->initializer);

    code += ctx->format(su::joinE(stmt->type_name, " ", stmt->var_name, " = ", init_str.code, ";\n"));
  }

  constexpr void walk(EmptyLineStatement* stmt) override {
    (void)stmt;
    code += "\n";
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
}

