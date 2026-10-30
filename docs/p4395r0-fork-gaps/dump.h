// Describe a statement/expression tree using whatever predicates the fork
// has, then print it through a failing static_assert message.
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
  try {
    add(sm::is_statement(r), "stmt");
  } catch (...) {
  }
  try {
    add(sm::is_expression(r), "expr");
  } catch (...) {
  }
  try {
    add(sm::is_return_statement(r), "return");
  } catch (...) {
  }
  try {
    add(sm::is_declaration_statement(r), "decl");
  } catch (...) {
  }
  try {
    add(sm::is_function_call(r), "call");
  } catch (...) {
  }
  try {
    add(sm::is_literal(r), "literal");
  } catch (...) {
  }
  try {
    add(sm::is_conditional_operator(r), "cond");
  } catch (...) {
  }
  try {
    add(sm::is_construct(r), "construct");
  } catch (...) {
  }
  try {
    add(sm::is_variable(r), "variable");
  } catch (...) {
  }
  try {
    add(sm::is_function(r), "function");
  } catch (...) {
  }
#if defined(_GLIBCXX_META)
  try {
    add(sm::is_unary_operator(r), "unary");
  } catch (...) {
  }
  try {
    add(sm::is_binary_operator(r), "binary");
  } catch (...) {
  }
  try {
    add(sm::is_cast(r), "cast");
  } catch (...) {
  }
#else
  try {
    add(sm::is_compound_statement(r), "compound");
  } catch (...) {
  }
  try {
    add(sm::is_expression_statement(r), "exprstmt");
  } catch (...) {
  }
  try {
    add(sm::is_if_statement(r), "if");
  } catch (...) {
  }
  try {
    add(sm::is_for_statement(r), "for");
  } catch (...) {
  }
  try {
    add(sm::is_simple_declaration(r), "simple");
  } catch (...) {
  }
  try {
    add(sm::is_id_expression(r), "id");
  } catch (...) {
  }
  try {
    add(sm::is_operator_expression(r), "op");
  } catch (...) {
  }
  try {
    add(sm::is_conversion(r), "conversion");
  } catch (...) {
  }
#endif
  bool expr = false;
  try {
    expr = sm::is_expression(r);
  } catch (...) {
  }
#if defined(_GLIBCXX_META)
  bool opx = expr;
  try {
    opx = expr && (sm::is_unary_operator(r) || sm::is_binary_operator(r) ||
                   sm::is_function_call(r));
  } catch (...) {
  }
#else
  bool opx = expr && sm::is_operator_expression(r);
#endif
  if (opx) {
    try {
      auto op = sm::expression_operator_of(r);
      s += "op=";
      s += sm::symbol_of(op);
      s += ' ';
    } catch (...) {
    }
  }
#ifdef SHOW_NAMES
  try {
    s += "'";
    s += sm::display_string_of(r);
    s += "' ";
  } catch (...) {
  }
#endif
  return s;
}

consteval std::string tree(sm::info r, int depth, int maxdepth) {
  std::string s(depth * 2, ' ');
  s += "[" + kinds(r) + "]\n";
  if (depth >= maxdepth)
    return s;
  bool expr = false, stmt = false;
  try {
    expr = sm::is_expression(r);
  } catch (...) {
  }
  try {
    stmt = sm::is_statement(r);
  } catch (...) {
  }
#if !defined(_GLIBCXX_META)
  // Follow the paper's accessors rather than statements_of on every node.
  if (sm::is_compound_statement(r)) {
    for (sm::info c : sm::statements_of(r))
      s += tree(c, depth + 1, maxdepth);
  } else if (sm::is_expression_statement(r)) {
    s += tree(sm::expression_of(r), depth + 1, maxdepth);
  } else if (sm::is_return_statement(r)) {
    s += tree(sm::return_operand_of(r), depth + 1, maxdepth);
  } else if (sm::is_declaration_statement(r) && sm::is_simple_declaration(r)) {
    auto d = sm::declarators_of(r);
    s +=
        std::string(depth * 2 + 2, ' ') + "declarators=" + num(d.size()) + "\n";
    for (sm::info i : sm::initializers_of(r))
      if (i != sm::info{})
        s += tree(i, depth + 1, maxdepth);
  } else if (sm::is_if_statement(r)) {
    s += tree(sm::then_statement_of(r), depth + 1, maxdepth);
    if (sm::else_statement_of(r) != sm::info{})
      s += tree(sm::else_statement_of(r), depth + 1, maxdepth);
  } else if (expr) {
    for (sm::info c : sm::operands_of(r))
      s += tree(c, depth + 1, maxdepth);
  }
  return s;
#endif
  if (stmt && (!expr || depth == 0)) {
    try {
      for (sm::info c : sm::statements_of(r))
        s += tree(c, depth + 1, maxdepth);
    } catch (...) {
    }
  }
  if (expr && depth > 0) {
    try {
      for (sm::info c : sm::operands_of(r))
        s += tree(c, depth + 1, maxdepth);
    } catch (...) {
    }
  }
  return s;
}

#define DUMP(fn, depth)                                                        \
  static_assert(false, std::string("\n=== " #fn "\n") +                        \
                           tree(sm::body_of(^^fn), 0, depth))
