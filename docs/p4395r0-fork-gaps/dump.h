// Describe a statement/expression tree with the paper's predicates, then print
// it through a failing static_assert message. The walk has a depth limit, so a
// fork that hands back a cycle can't exhaust memory.
#include <meta>
#include <string>
#include <vector>
namespace sm = std::meta;

consteval std::string num(std::size_t n) {
  std::string s;
  do {
    s.insert(s.begin(), char('0' + n % 10));
    n /= 10;
  } while (n);
  return s;
}

consteval std::string kinds(sm::info r) {
  std::string s;
  auto add = [&](bool b, const char *n) {
    if (b) {
      s += n;
      s += ' ';
    }
  };
  if (sm::is_statement(r)) {
    add(true, "stmt");
    add(sm::is_compound_statement(r), "compound");
    add(sm::is_declaration_statement(r), "decl");
    add(sm::is_expression_statement(r), "exprstmt");
    add(sm::is_return_statement(r), "return");
#ifdef __clang__
    add(sm::is_if_statement(r), "if");
    add(sm::is_for_statement(r), "for");
#endif
    if (sm::is_declaration_statement(r))
      add(sm::is_simple_declaration(r), "simple");
  }
  if (sm::is_expression(r)) {
    add(true, "expr");
    add(sm::is_id_expression(r), "id");
    add(sm::is_literal(r), "literal");
    add(sm::is_function_call(r), "call");
    add(sm::is_operator_expression(r), "op");
    add(sm::is_conditional_operator(r), "cond");
    add(sm::is_conversion(r), "conversion");
    add(sm::is_construct(r), "construct");
    if (sm::is_operator_expression(r)) {
      s += "op=";
      s += sm::symbol_of(sm::expression_operator_of(r));
      s += ' ';
    }
  }
  return s;
}

consteval std::string tree(sm::info r, int depth, int maxdepth) {
  std::string s(depth * 2, ' ');
  s += "[" + kinds(r) + "]\n";
  if (depth >= maxdepth)
    return s;
  if (sm::is_statement(r)) {
    if (sm::is_compound_statement(r)) {
      for (sm::info c : sm::statements_of(r))
        s += tree(c, depth + 1, maxdepth);
    } else if (sm::is_expression_statement(r)) {
      s += tree(sm::expression_of(r), depth + 1, maxdepth);
    } else if (sm::is_return_statement(r)) {
      s += tree(sm::return_operand_of(r), depth + 1, maxdepth);
    } else if (sm::is_declaration_statement(r) &&
               sm::is_simple_declaration(r)) {
      s += std::string(depth * 2 + 2, ' ') +
           "declarators=" + num(sm::declarators_of(r).size()) + "\n";
      for (sm::info i : sm::initializers_of(r))
        if (i != sm::info{})
          s += tree(i, depth + 1, maxdepth);
    }
#ifdef __clang__
    else if (sm::is_if_statement(r)) {
      s += tree(sm::then_statement_of(r), depth + 1, maxdepth);
      if (sm::else_statement_of(r) != sm::info{})
        s += tree(sm::else_statement_of(r), depth + 1, maxdepth);
    }
#endif
  } else if (sm::is_expression(r)) {
    for (sm::info c : sm::operands_of(r))
      s += tree(c, depth + 1, maxdepth);
  }
  return s;
}

#define DUMP(fn, depth)                                                        \
  static_assert(false, std::string("\n=== " #fn "\n") +                        \
                           tree(sm::body_of(^^fn), 0, depth))
