#pragma once

#include <string>

#include "../autograd.h"
#include "../meta.h"

namespace ad {
using std::meta::info;
namespace m = ad::meta;

enum class ExpressionKind {
  Literal,
  Id,
  FunctionCall,
  Operator
};

struct Expression {
  ExpressionKind kind;

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

  consteval IdExpression(info expr) : Expression(ExpressionKind::Id),
    id(m::identifier_of(m::declaration_of(expr))) {}

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
    }
  }

  constexpr ~OperatorExpression() {
    for (Expression* expr : operands) {
      freeExpression(expr);
    }
  }

  DECL_BODY(OperatorExpression)

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    for (Expression* expr : operands) {
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
    }
  }

  constexpr ~FunctionCallExpression() {
    for (Expression* expr : operands) {
      freeExpression(expr);
    }
  }

  DECL_BODY(FunctionCallExpression)

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    for (Expression* expr : operands) {
      walker->walk(expr);
    }
  }
};

consteval Expression* expressionFromInfo(info e) {
  info expr = ad::detail::stripCasts(e); // TODO: Handle casts.

  if (m::is_operator_expression(expr)) {
    return new OperatorExpression(expr);
  } else if (m::is_literal(expr)) {
    return new LiteralExpression(expr);
  } else if (m::is_id_expression(expr)) {
    return new IdExpression(expr);
  } else if (m::is_function_call(expr)) {
    return new FunctionCallExpression(expr);
  } else {
    throwKind(expr);
    return nullptr;
  }
}

constexpr void freeExpression(Expression*& expr) {
  if (expr->isOperator()) {
    delete OperatorExpression::cast(expr);
  } else if (expr->isLiteral()) {
    delete LiteralExpression::cast(expr);
  } else if (expr->isFunctionCall()) {
    delete FunctionCallExpression::cast(expr);
  } else if (expr->isId()) {
    delete IdExpression::cast(expr);
  } else {
    throw "Unknown expression";
  }

  expr = nullptr;
}



#undef DECL_BODY

enum class StatementKind {
  Declaration,
  Expression,
  Return
};

struct Statement {
  StatementKind kind;

  constexpr bool isDeclaration() {
    return StatementKind::Declaration == kind;
  }

  constexpr bool isExpression() {
    return StatementKind::Expression == kind;
  }

  constexpr bool isReturn() {
    return StatementKind::Return == kind;
  }

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    (void)walker;
  }
};

struct DeclarationStatement : public Statement {

  std::string type_name;
  std::string var_name;

  Expression* initializer;

  consteval DeclarationStatement(info stmt) : Statement(StatementKind::Declaration),
    type_name(m::display_string_of(m::type_of(m::declared_variable_of(stmt)))),
    var_name(m::identifier_of(m::declared_variable_of(stmt))),
    initializer(expressionFromInfo(initializer_of(m::declared_variable_of(stmt)))) {}

  constexpr ~DeclarationStatement() {
    freeExpression(initializer);
  }

  static constexpr DeclarationStatement* cast(Statement* stmt) {
    return static_cast<DeclarationStatement*>(stmt);
  }

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    walker->walk(initializer);
  }
};

struct ExpressionStatement : public Statement {

  Expression* expression;

  consteval ExpressionStatement(info stmt) : Statement(StatementKind::Expression),
    expression(expressionFromInfo(m::expression_of(stmt))) {}

  static constexpr ExpressionStatement* cast(Statement* stmt) {
    return static_cast<ExpressionStatement*>(stmt);
  }

  constexpr ~ExpressionStatement() {
    freeExpression(expression);
  }

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    walker->walk(expression);
  }
};

struct ReturnStatement : public Statement {

  Expression* return_expression;

  consteval ReturnStatement(info stmt) : Statement(StatementKind::Return),
    return_expression(expressionFromInfo(m::return_operand_of(stmt)))
  { (void)stmt; }

  static constexpr ReturnStatement* cast(Statement* stmt) {
    return static_cast<ReturnStatement*>(stmt);
  }

  constexpr ~ReturnStatement() {
    freeExpression(return_expression);
  }

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    walker->walk(return_expression);
  }
};

constexpr void releaseStatement(Statement*& stmt) {
  if (stmt->isDeclaration()) {
    delete DeclarationStatement::cast(stmt);
  }
  else if (stmt->isExpression()) {
    delete ExpressionStatement::cast(stmt);
  }
  else if (stmt->isReturn()) {
    delete ReturnStatement::cast(stmt);
  }
  else {
    throw "Unknown expression.";
  }

  stmt = nullptr;
}

struct AstWalker {

#define DECL_HANDLE(type, name) \
  constexpr virtual void handle(type* name) { \
    (void)name; \
  } \
  constexpr virtual void walk(type* name) { \
    handle(name); \
    name->walk(this); \
  }

  DECL_HANDLE(IdExpression, expr)
  DECL_HANDLE(FunctionCallExpression, expr)
  DECL_HANDLE(OperatorExpression, expr)
  DECL_HANDLE(LiteralExpression, expr)

  DECL_HANDLE(DeclarationStatement, stmt)
  DECL_HANDLE(ExpressionStatement, stmt)
  DECL_HANDLE(ReturnStatement, stmt)

#undef DECL_HANDLE

  constexpr void walk(Expression* expr) {
    if (expr->isOperator()) {
      walk(OperatorExpression::cast(expr));
    } else if (expr->isLiteral()) {
      walk(LiteralExpression::cast(expr));
    } else if (expr->isFunctionCall()) {
      walk(FunctionCallExpression::cast(expr));
    } else if (expr->isId()) {
      walk(IdExpression::cast(expr));
    } else {
      throw "Unknown expression";
    }

    expr = nullptr;
  }

  constexpr virtual void walk(std::vector<Statement*>* stmts) {
    for (Statement*& stmt : *stmts) {
      if (stmt->isDeclaration()) {
        walk(DeclarationStatement::cast(stmt));
      }
      else if (stmt->isExpression()) {
        walk(ExpressionStatement::cast(stmt));
      }
      else if (stmt->isReturn()) {
        walk(ReturnStatement::cast(stmt));
      }
      else {
        throw "Unknown expression.";
      }
    }
  }
};

consteval std::vector<Statement*> fromFunctionBody(info body) {
  std::vector<Statement*> r = {};

  for (info stmt : m::statements_of(body)) {
    if (m::is_declaration_statement(stmt)) {
      r.push_back(new DeclarationStatement(stmt));
    } else if (m::is_expression_statement(stmt)) {
      r.push_back(new ExpressionStatement(stmt));
    } else if (m::is_return_statement(stmt)) {
      r.push_back(new ReturnStatement(stmt));
    } else {
      throwKind(stmt);
    }
  }

  return r;
}

constexpr void releaseFunctionBody(std::vector<Statement*>& body) {
  for (Statement*& stmt : body) {
    releaseStatement(stmt);
  }
}
}