// smtlib.hpp — translate a reflected function body into SMT-LIB.
//
// smt::define_fun(^^f) gives an SMT-LIB `define-fun` with the same meaning as
// f, for a loop-free f over `int`: declarations, assignments, if statements
// (with or without else), `?:`, return, `+ - *`, unary `-`, comparisons and
// `&& || !`. An int is a 32-bit bitvector, so arithmetic wraps the way the
// hardware does, which is what lets a solver find an overflow. Anything else
// throws, so an unsupported body fails to compile instead of producing a
// formula that means something else.
//
// The body is executed symbolically. Each assignment binds a fresh name in a
// `let`; an if statement runs both branches and merges every variable, and
// the return value, with an `ite` on the condition. A return is tracked per
// path: `returned` is the condition under which the function has returned so
// far, and later assignments and returns only apply where it is false.
//
// See docs/smt-verification.md for how to check the output with a solver.

#ifndef SMT_SMTLIB_HPP
#define SMT_SMTLIB_HPP

#include <ad/meta.h>
#include <string>
#include <utility>
#include <vector>

namespace smt {

namespace m = ad::meta;

namespace detail {

struct term {
  std::string text;
  bool is_bool;
};

struct binding {
  std::string name;
  std::string text;
};

struct state {
  std::vector<std::pair<std::string, std::string>> env; // C++ name -> symbol
  std::string ret = "(_ bv0 32)";
  std::string returned = "false";
};

// A function's lets, in order, and the counter that keeps their names fresh.
struct context {
  std::vector<binding> lets;
  int fresh = 0;
};

consteval std::string bv(int v) {
  // SMT-LIB has no negative literals; bvneg of the magnitude would overflow
  // for INT_MIN, so spell out the two's complement bits instead.
  unsigned u = static_cast<unsigned>(v);
  const char *hex = "0123456789abcdef";
  std::string r = "#x";
  for (int shift = 28; shift >= 0; shift -= 4)
    r += hex[(u >> shift) & 0xf];
  return r;
}

consteval std::string as_bool(const term &t) {
  return t.is_bool ? t.text : "(not (= " + t.text + " #x00000000))";
}

consteval std::string as_bv(const term &t) {
  return t.is_bool ? "(ite " + t.text + " #x00000001 #x00000000)" : t.text;
}

consteval std::string lookup(const state &s, const std::string &name) {
  for (auto it = s.env.rbegin(); it != s.env.rend(); ++it)
    if (it->first == name)
      return it->second;
  throw "smt: use of an unknown variable";
}

consteval void set(state &s, const std::string &name, std::string symbol) {
  for (auto &e : s.env)
    if (e.first == name) {
      e.second = std::move(symbol);
      return;
    }
  s.env.push_back({name, std::move(symbol)});
}

// Bind TEXT to a fresh name for NAME and return the name.
consteval std::string bind(context &c, const std::string &name,
                           std::string text) {
  std::string symbol = name + "!" + std::to_string(++c.fresh);
  c.lets.push_back({symbol, std::move(text)});
  return symbol;
}

consteval m::info strip(m::info e) {
  while (m::is_conversion(e))
    e = m::operands_of(e).front();
  return e;
}

consteval std::string name_of(m::info id) {
  return std::string(m::identifier_of(m::declaration_of(id)));
}

consteval bool is_assignment(m::operators op) {
  return op == m::operators::op_equals || op == m::operators::op_plus_equals ||
         op == m::operators::op_minus_equals ||
         op == m::operators::op_star_equals;
}

consteval term expr(m::info e, const state &s);

consteval term binary(m::operators op, const term &l, const term &r) {
  using enum m::operators;
  auto bvop = [&](const char *f) {
    return term{std::string("(") + f + " " + as_bv(l) + " " + as_bv(r) + ")",
                false};
  };
  auto cmp = [&](const char *f) {
    return term{std::string("(") + f + " " + as_bv(l) + " " + as_bv(r) + ")",
                true};
  };
  auto logic = [&](const char *f) {
    return term{
        std::string("(") + f + " " + as_bool(l) + " " + as_bool(r) + ")", true};
  };
  if (op == op_plus || op == op_plus_equals)
    return bvop("bvadd");
  if (op == op_minus || op == op_minus_equals)
    return bvop("bvsub");
  if (op == op_star || op == op_star_equals)
    return bvop("bvmul");
  if (op == op_less)
    return cmp("bvslt");
  if (op == op_less_equals)
    return cmp("bvsle");
  if (op == op_greater)
    return cmp("bvsgt");
  if (op == op_greater_equals)
    return cmp("bvsge");
  if (op == op_equals_equals)
    return cmp("=");
  if (op == op_exclamation_equals)
    return term{"(not (= " + as_bv(l) + " " + as_bv(r) + "))", true};
  if (op == op_ampersand_ampersand)
    return logic("and");
  if (op == op_pipe_pipe)
    return logic("or");
  throw "smt: unsupported binary operator";
}

consteval term expr(m::info e, const state &s) {
  e = strip(e);
  if (m::is_literal(e))
    return {bv(m::extract<int>(m::constant_of(e))), false};
  if (m::is_id_expression(e))
    return {lookup(s, name_of(e)), false};
  if (m::is_conditional_operator(e)) {
    term c = expr(m::condition_of(e), s);
    term t = expr(m::true_expression_of(e), s);
    term f = expr(m::false_expression_of(e), s);
    if (t.is_bool && f.is_bool)
      return {"(ite " + as_bool(c) + " " + t.text + " " + f.text + ")", true};
    return {"(ite " + as_bool(c) + " " + as_bv(t) + " " + as_bv(f) + ")",
            false};
  }
  if (m::is_operator_expression(e)) {
    const m::operators op = m::expression_operator_of(e);
    const std::vector<m::info> ops = m::operands_of(e);
    if (ops.size() == 1) {
      term a = expr(ops[0], s);
      if (op == m::operators::op_minus)
        return {"(bvneg " + as_bv(a) + ")", false};
      if (op == m::operators::op_plus)
        return {as_bv(a), false};
      if (op == m::operators::op_exclamation)
        return {"(not " + as_bool(a) + ")", true};
      throw "smt: unsupported unary operator";
    }
    if (ops.size() == 2 && !is_assignment(op))
      return binary(op, expr(ops[0], s), expr(ops[1], s));
  }
  throw "smt: unsupported expression";
}

// Where the function has already returned, keep OLD; elsewhere take NEW.
consteval std::string guarded(const state &s, const std::string &old_value,
                              const std::string &new_value) {
  if (s.returned == "false")
    return new_value;
  return "(ite " + s.returned + " " + old_value + " " + new_value + ")";
}

consteval void assign(context &c, state &s, const std::string &name,
                      const term &value) {
  std::string old_value = "(_ bv0 32)";
  for (auto &e : s.env)
    if (e.first == name)
      old_value = e.second;
  set(s, name, bind(c, name, guarded(s, old_value, as_bv(value))));
}

consteval void stmt(context &c, state &s, m::info st);

consteval state merge(context &c, const std::string &cond, const state &before,
                      const state &a, const state &b) {
  state r = before;
  for (const auto &e : a.env) {
    std::string other;
    bool found = false;
    for (const auto &f : b.env)
      if (f.first == e.first) {
        other = f.second;
        found = true;
      }
    if (!found)
      continue; // declared in one branch only: out of scope after the if
    if (e.second == other)
      set(r, e.first, e.second);
    else
      set(r, e.first,
          bind(c, e.first,
               "(ite " + cond + " " + e.second + " " + other + ")"));
  }
  // Locals declared inside a branch go out of scope.
  std::vector<std::pair<std::string, std::string>> scoped;
  for (const auto &e : r.env)
    for (const auto &f : before.env)
      if (f.first == e.first)
        scoped.push_back(e);
  r.env = scoped;
  r.ret =
      a.ret == b.ret
          ? a.ret
          : bind(c, "ret", "(ite " + cond + " " + a.ret + " " + b.ret + ")");
  r.returned =
      a.returned == b.returned
          ? a.returned
          : bind(c, "returned",
                 "(ite " + cond + " " + a.returned + " " + b.returned + ")");
  return r;
}

// An if statement's branches. GCC reflects an if as a conditional, whose
// missing else is a void conversion; clang has then/else accessors.
consteval m::info then_of(m::info st) {
#if defined(_GLIBCXX_META)
  return m::true_expression_of(st);
#else
  return m::then_statement_of(st);
#endif
}

consteval m::info else_of(m::info st) {
#if defined(_GLIBCXX_META)
  return m::false_expression_of(st);
#else
  return m::else_statement_of(st);
#endif
}

// An assignment `x = e` (or `x op= e`), as an expression.
consteval bool assignment(context &c, state &s, m::info e) {
  e = strip(e);
  if (!m::is_operator_expression(e) ||
      !is_assignment(m::expression_operator_of(e)))
    return false;
  const auto ops = m::operands_of(e);
  const std::string name = name_of(strip(ops[0]));
  term value = expr(ops[1], s);
  const m::operators op = m::expression_operator_of(e);
  if (op != m::operators::op_equals)
    value = binary(op, term{lookup(s, name), false}, value);
  assign(c, s, name, value);
  return true;
}

consteval void stmt(context &c, state &s, m::info st) {
  if (st == m::info{})
    return;
  if (m::is_compound_statement(st)) {
    for (m::info k : m::statements_of(st))
      stmt(c, s, k);
    return;
  }
  if (m::is_return_statement(st)) {
    term v = expr(m::return_operand_of(st), s);
    const std::string rv = as_bv(v);
    s.ret = bind(c, "ret", guarded(s, s.ret, rv));
    s.returned = "true";
    return;
  }
  if (m::is_declaration_statement(st)) {
    for (m::info var : m::declarators_of(st)) {
      m::info init = m::initializer_of(var);
      if (init == m::info{})
        throw "smt: a declaration needs an initializer";
      assign(c, s, std::string(m::identifier_of(var)), expr(init, s));
    }
    return;
  }
  if (m::is_expression_statement(st)) {
    if (!assignment(c, s, m::expression_of(st)))
      throw "smt: unsupported expression statement";
    return;
  }
  if (m::is_if_statement(st)) {
    const std::string cond = as_bool(expr(m::condition_of(st), s));
    state a = s;
    stmt(c, a, then_of(st));
    state b = s;
    stmt(c, b, else_of(st));
    s = merge(c, cond, s, a, b);
    return;
  }
  // GCC gives an if's branch as an expression when it is one, and a missing
  // else as a void conversion.
  if (m::is_expression(st)) {
    if (assignment(c, s, st) || m::is_conversion(st))
      return;
    throw "smt: unsupported expression in a branch";
  }
  throw "smt: unsupported statement";
}

} // namespace detail

// `(define-fun NAME ((p (_ BitVec 32)) ...) (_ BitVec 32) BODY)` for a
// function over int.
consteval std::string define_fun(m::info fn) {
  detail::context c;
  detail::state s;
  std::string params;
  for (m::info p : m::parameters_of(fn)) {
    const std::string name(m::identifier_of(p));
    params += (params.empty() ? "(" : " (") + name + " (_ BitVec 32))";
    s.env.push_back({name, name});
  }
  detail::stmt(c, s, m::body_of(fn));
  std::string body = s.ret;
  for (auto it = c.lets.rbegin(); it != c.lets.rend(); ++it)
    body = "(let ((" + it->name + " " + it->text + "))\n  " + body + ")";
  return "(define-fun " + std::string(m::identifier_of(fn)) + " (" + params +
         ") (_ BitVec 32)\n  " + body + ")\n";
}

} // namespace smt

#endif
