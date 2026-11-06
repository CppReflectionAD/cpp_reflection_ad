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

  constexpr Expression* clone() override = 0;
};

#define DECL_BODY(name) \
static constexpr name* cast(Expression* expr) { \
  return static_cast<name*>(expr); \
}

consteval Expression* expressionFromInfo(info expr);
constexpr std::vector<Expression*> cloneExpressionVec(std::vector<Expression*> const& vec);
constexpr void freeExpression(Expression*& expr);

consteval std::vector<Expression*> expressionsFromInfoVec(auto&& vec) {
  std::vector<Expression*> r = {};
  r.reserve(vec.size());

  for (info cur : vec) {
    r.push_back(expressionFromInfo(cur));
  }

  return r;
}


struct LiteralExpression : public Expression {

  std::string literal;

  constexpr LiteralExpression(std::string_view const& literal) : Expression(ExpressionKind::Literal),
    literal(literal) {}

  consteval LiteralExpression(info expr) : LiteralExpression(m::display_string_of(m::constant_of(expr))) {}

  DECL_BODY(LiteralExpression)

  constexpr LiteralExpression* clone() override {
    return new LiteralExpression(literal);
  }
};

struct IdExpression : public Expression {

  std::string id;

  constexpr IdExpression(std::string_view const& id) : Expression(ExpressionKind::Id), id(id) {}

  consteval IdExpression(info expr) : IdExpression(m::identifier_of(m::declaration_of(expr))) {}

  DECL_BODY(IdExpression)

  constexpr IdExpression* clone() override {
    return new IdExpression(id);
  }
};

struct OperatorExpression : public Expression {

  std::vector<Expression*> operands;
  m::operators op;

  constexpr OperatorExpression(m::operators op, std::vector<Expression*> const& ops) :
    Expression(ExpressionKind::Operator), operands(ops), op(op) {
    for (Expression* operand : operands) {
      operand->parent = this;
    }
  }

  consteval OperatorExpression(info expr) : OperatorExpression(
    m::expression_operator_of(expr),
    expressionsFromInfoVec(m::operands_of(expr))) {}

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

  constexpr OperatorExpression* clone() override {
    return new OperatorExpression(op, cloneExpressionVec(operands));
  }
};

struct FunctionCallExpression : public Expression {

  std::vector<Expression*> operands;
  std::string func_name;

  constexpr FunctionCallExpression(std::string_view const& func_name, std::vector<Expression*> const& ops) :
    Expression(ExpressionKind::FunctionCall), operands(ops), func_name(func_name) {
    for (Expression* operand : operands) {
      operand->parent = this;
    }
  }

  consteval FunctionCallExpression(info expr) : FunctionCallExpression(
    m::identifier_of(m::declaration_of(m::callee_of(expr))),
    expressionsFromInfoVec(m::operands_of(expr))) {}

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

  constexpr FunctionCallExpression* clone() override {
    return new FunctionCallExpression(func_name, cloneExpressionVec(operands));
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

constexpr std::vector<Expression*> cloneExpressionVec(std::vector<Expression*> const& vec) {
  std::vector<Expression*> r = {};
  r.reserve(vec.size());

  for (Expression* cur : vec) {
    r.push_back(cur->clone());
  }

  return r;
}

constexpr void freeExpression(Expression*& expr) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdelete-non-abstract-non-virtual-dtor"

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
#pragma GCC diagnostic pop
}

#undef DECL_BODY
}