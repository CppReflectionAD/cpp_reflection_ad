#pragma once

#include <string>

#include "../../autograd.h"
#include "../../meta.h"
#include "../throw_helper.hpp"
#include "ast_item.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;

enum class ExpressionKind {
  Literal,
  Id,
  FunctionCall,
  Operator
};

struct Expression : public ASTItem {
  ExpressionKind kind;

  constexpr Expression(ExpressionKind kind) : ASTItem(ASTKind::Expression), kind(kind) {}

#define DECL_IS(name) \
  constexpr bool is ## name() { \
    return ExpressionKind:: name == kind; \
  }

  DECL_IS(Literal)
  DECL_IS(Id)
  DECL_IS(FunctionCall)
  DECL_IS(Operator)

#undef DECL_IS

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    (void)walker;
  }

  constexpr static Expression* cast(ASTItem* item) {
    return static_cast<Expression*>(item);
  }
};

#define DECL_BODY(name) \
static constexpr name* cast(Expression* expr) { \
  return static_cast<name*>(expr); \
}

consteval Expression* expressionFromInfo(info expr);
constexpr void freeExpression(Expression*& expr);


struct LiteralExpression : public Expression {

  std::string literal;

  consteval LiteralExpression(info expr) : Expression(ExpressionKind::Literal),
    literal(m::display_string_of(m::constant_of(expr))) {}

  DECL_BODY(LiteralExpression)
};

struct IdExpression : public Expression {

  std::string id;

  constexpr IdExpression(std::string_view const& id) : Expression(ExpressionKind::Id), id(id) {}

  consteval IdExpression(info expr) : IdExpression(m::identifier_of(m::declaration_of(expr))) {}

  DECL_BODY(IdExpression)
};

struct OperatorExpression : public Expression {

  std::vector<Expression*> operands;
  m::operators op;

  consteval OperatorExpression(info expr) : Expression(ExpressionKind::Operator),
    operands(),
    op(m::expression_operator_of(expr)) {
    for (info operand : m::operands_of(expr)) {
      operands.push_back(expressionFromInfo(operand));
      operands.back()->parent = this;
    }
  }

  constexpr ~OperatorExpression() {
    for (Expression*& expr : operands) {
      freeExpression(expr);
    }
  }

  DECL_BODY(OperatorExpression)

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    for (Expression*& expr : operands) {
      walker->walk(expr);
    }
  }
};

struct FunctionCallExpression : public Expression {

  std::vector<Expression*> operands;
  std::string func_name;

  consteval FunctionCallExpression(info expr) : Expression(ExpressionKind::FunctionCall),
    operands(),
    func_name(m::identifier_of(m::declaration_of(m::callee_of(expr)))) {
    for (info operand : m::operands_of(expr)) {
      operands.push_back(expressionFromInfo(operand));
      operands.back()->parent = this;
    }
  }

  constexpr ~FunctionCallExpression() {
    for (Expression*& expr : operands) {
      freeExpression(expr);
    }
  }

  DECL_BODY(FunctionCallExpression)

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    for (Expression*& expr : operands) {
      walker->walk(expr);
    }
  }
};

consteval Expression* expressionFromInfo(info e) {
  info expr = ad::detail::stripCasts(e); // TODO: Handle casts.

  if (m::is_operator_expression(expr)) {
    return new OperatorExpression(expr);
  }
  else if (m::is_literal(expr)) {
    return new LiteralExpression(expr);
  }
  else if (m::is_id_expression(expr)) {
    return new IdExpression(expr);
  }
  else if (m::is_function_call(expr)) {
    return new FunctionCallExpression(expr);
  }
  else {
    throwKind(expr);
    return nullptr;
  }
}

constexpr void freeExpression(Expression*& expr) {
  if (expr->isOperator()) {
    delete OperatorExpression::cast(expr);
  }
  else if (expr->isLiteral()) {
    delete LiteralExpression::cast(expr);
  }
  else if (expr->isFunctionCall()) {
    delete FunctionCallExpression::cast(expr);
  }
  else if (expr->isId()) {
    delete IdExpression::cast(expr);
  }
  else {
    throw "Unknown expression";
  }

  expr = nullptr;
}

#undef DECL_BODY
}