#pragma once

#include <string>

#include "../../autograd.h"
#include "../../meta.h"
#include "../throw_helper.hpp"
#include "ast_item.hpp"
#include "expressions.hpp"

namespace ad {

using std::meta::info;
namespace m = ad::meta;

enum class StatementKind {
  Comment,
  Declaration,
  EmptyLine,
  Expression,
  Return
};

template<StatementKind k>
struct StatementKindToClass {
  using type = NoType;
};

template<StatementKind k>
using StatementKindToClass_t = typename StatementKindToClass<k>::type;

struct Statement : public ASTItem {
  StatementKind kind;

  explicit constexpr Statement(StatementKind kind) : ASTItem(ASTKind::Statement), kind(kind) {}

  constexpr bool is(StatementKind k) const {
    return k == kind;
  }

  constexpr static Statement* cast(ASTItem* item) {
    return static_cast<Statement*>(item);
  }

  template<typename Walker>
  constexpr void walk(Walker* walker) {
    (void)walker;
  }

  constexpr Statement* clone() override = 0;
};

struct CommentStatement : public Statement {
  std::string comment;

  constexpr CommentStatement(std::string const& comment) : Statement(StatementKind::Comment), comment(comment) {}

  static constexpr CommentStatement* cast(Statement* stmt) {
    return static_cast<CommentStatement*>(stmt);
  }

  constexpr CommentStatement* clone() override {
    return new CommentStatement(comment);
  }
};
template <> struct StatementKindToClass<StatementKind::Comment> { using type = CommentStatement; };

struct DeclarationStatement : public Statement {

  std::string type_name;
  std::string var_name;

  Expression* initializer;

  constexpr DeclarationStatement(std::string_view const& type_name, std::string_view const& var_name, Expression* init) :
    Statement(StatementKind::Declaration),
      type_name(type_name),
      var_name(var_name),
      initializer(init) {
    initializer->parent = this;
  }

  consteval DeclarationStatement(info stmt) : DeclarationStatement(
      m::display_string_of(m::type_of(m::declared_variable_of(stmt))),
      m::identifier_of(m::declared_variable_of(stmt)),
      expressionFromInfo(initializer_of(m::declared_variable_of(stmt)))) {}

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

  constexpr DeclarationStatement* clone() override {
    return new DeclarationStatement(type_name, var_name, initializer->clone());
  }
};
template <> struct StatementKindToClass<StatementKind::Declaration> { using type = DeclarationStatement; };

struct EmptyLineStatement : public Statement {

  constexpr EmptyLineStatement() : Statement(StatementKind::EmptyLine) {}

  static constexpr EmptyLineStatement* cast(Statement* stmt) {
    return static_cast<EmptyLineStatement*>(stmt);
  }

  constexpr EmptyLineStatement* clone() override {
    return new EmptyLineStatement();
  }
};
template <> struct StatementKindToClass<StatementKind::EmptyLine> { using type = EmptyLineStatement; };

struct ExpressionStatement : public Statement {

  Expression* expression;

  constexpr ExpressionStatement(Expression* expr) : Statement(StatementKind::Expression),
    expression(expr) {
    expression->parent = this;
  }

  consteval ExpressionStatement(info stmt) : ExpressionStatement(expressionFromInfo(m::expression_of(stmt))) {}

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

  constexpr ExpressionStatement* clone() override {
    return new ExpressionStatement(expression->clone());
  }
};
template <> struct StatementKindToClass<StatementKind::Expression> { using type = ExpressionStatement; };

struct ReturnStatement : public Statement {

  Expression* return_expression;

  constexpr ReturnStatement(Expression* expr) : Statement(StatementKind::Return),
  return_expression(expr) {
    return_expression->parent = this;
  }

  consteval ReturnStatement(info stmt) : ReturnStatement(expressionFromInfo(m::return_operand_of(stmt))) {}

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

  constexpr ReturnStatement* clone() override {
    return new ReturnStatement(return_expression->clone());
  }
};
template <> struct StatementKindToClass<StatementKind::Return> { using type = ReturnStatement; };

constexpr void releaseStatement(Statement*& stmt) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdelete-non-abstract-non-virtual-dtor"

  constexpr static auto enumerators = std::define_static_array(enumerators_of(^^StatementKind));
  template for (constexpr auto enumerator : enumerators) {
    if (nullptr != stmt && stmt->is([:enumerator:])) {
      delete StatementKindToClass_t<([:enumerator:])>::cast(stmt);
      stmt = nullptr;
      return;
    }
  }

  throw "Unknown expression";

#pragma GCC diagnostic pop
}

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