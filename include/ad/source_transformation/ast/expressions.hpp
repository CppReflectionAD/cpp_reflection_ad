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

template<ExpressionKind k>
struct ExpressionKindToClass {
  using type = NoType;
};

template<ExpressionKind k>
using ExpressionKindToClass_t = typename ExpressionKindToClass<k>::type;

struct Expression : public ASTItem {
  ExpressionKind kind;

  constexpr Expression(ExpressionKind kind) : ASTItem(ASTKind::Expression), kind(kind) {}

  constexpr bool is(ExpressionKind k) const {
    return k == kind;
  }

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    (void)walker;
  }

  constexpr static Expression* cast(ASTItem* item) {
    return static_cast<Expression*>(item);
  }

  constexpr Expression* clone() override = 0;
};

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

  static constexpr LiteralExpression* cast(Expression* expr) {
    return static_cast<LiteralExpression*>(expr);
  }

  constexpr LiteralExpression* clone() override {
    return new LiteralExpression(literal);
  }
};
template <> struct ExpressionKindToClass<ExpressionKind::Literal> { using type = LiteralExpression; };

struct IdExpression : public Expression {

  std::string id;

  constexpr IdExpression(std::string_view const& id) : Expression(ExpressionKind::Id), id(id) {}

  consteval IdExpression(info expr) : IdExpression(m::identifier_of(m::declaration_of(expr))) {}

  static constexpr IdExpression* cast(Expression* expr) {
    return static_cast<IdExpression*>(expr);
  }

  constexpr IdExpression* clone() override {
    return new IdExpression(id);
  }
};
template <> struct ExpressionKindToClass<ExpressionKind::Id> { using type = IdExpression; };

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

  static constexpr OperatorExpression* cast(Expression* expr) {
    return static_cast<OperatorExpression*>(expr);
  }

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
template <> struct ExpressionKindToClass<ExpressionKind::Operator> { using type = OperatorExpression; };

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

  static constexpr FunctionCallExpression* cast(Expression* expr) {
    return static_cast<FunctionCallExpression*>(expr);
  }

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
template <> struct ExpressionKindToClass<ExpressionKind::FunctionCall> { using type = FunctionCallExpression; };

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

  constexpr static auto enumerators = std::define_static_array(enumerators_of(^^ExpressionKind));
  template for (constexpr auto enumerator : enumerators) {
    if (nullptr != expr && expr->is([:enumerator:])) {
      delete ExpressionKindToClass_t<([:enumerator:])>::cast(expr);
      expr = nullptr;
      return;
    }
  }

  throw "Unknown expression";

#pragma GCC diagnostic pop
}

struct Wrap {
  Expression* wrap;

  constexpr Wrap operator=(this Wrap&& self, Wrap&& o) noexcept {
    return {new OperatorExpression(m::operators::op_equals, {self.wrap, o.wrap})};
  }

  constexpr Wrap operator+=(this Wrap&& self, Wrap&& o) noexcept {
    return {new OperatorExpression(m::operators::op_plus_equals, {self.wrap, o.wrap})};
  }
};

constexpr Wrap wrap(Expression* expr) {
  return {expr};
}

constexpr Wrap operator+(Wrap arg1, Wrap arg2) {
  return {new OperatorExpression(m::operators::op_plus, {arg1.wrap, arg2.wrap})};
}

constexpr Wrap operator*(Wrap arg1, Wrap arg2) {
  return {new OperatorExpression(m::operators::op_star, {arg1.wrap, arg2.wrap})};
}
}