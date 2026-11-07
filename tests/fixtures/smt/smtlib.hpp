// smtlib.hpp — translate a reflected function body into SMT-LIB.
//
// smt::define_fun(^^f) gives an SMT-LIB `define-fun` with the same meaning as
// f, for an f over `int`: declarations, assignments (`=`, `+=`, `-=`, `*=`,
// `++`, `--`), if statements (with or without else), `?:`, return, `+ - *`,
// unary `-`, comparisons and `&& || !`. An int is a 32-bit bitvector, so
// arithmetic wraps the way the hardware does, which is what lets a solver find
// an overflow. Anything else throws, so an unsupported body fails to compile
// instead of producing a formula that means something else.
//
// The body is executed symbolically. Each assignment binds a fresh name in a
// `let`; an if statement runs both branches and merges every variable, and
// the return value, with an `ite` on the condition. A return is tracked per
// path: `returned` is the condition under which the function has returned so
// far, and later assignments and returns only apply where it is false.
//
// Loops (while, do/while, for, with break and continue) need an unrolling
// bound: smt::define_fun(^^f, k) unrolls each loop k times and also defines
// `f_unwound`, which is true when k iterations were enough, so that a claim
// proved together with f_unwound holds for every run. GCC lowers a loop to
// labels, gotos and `if (cond) goto body; else goto exit;` before reflection
// sees it; recognizing that needs is_goto_statement, is_label_statement and
// label_of, which smt::loops_supported says the compiler has. Within a loop's
// body a goto to the exit label is a break, and one to a later label (a
// continue) skips the statements up to that label.
//
// See docs/smt-verification.md for how to check the output with a solver.

#ifndef SMT_SMTLIB_HPP
#define SMT_SMTLIB_HPP

#include <ad/meta.h>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace smt {

namespace m = ad::meta;

namespace detail {

// Found by argument-dependent lookup in std::meta, if it has them.
template <class I>
concept has_jump_api = requires(I i) {
  is_goto_statement(i);
  is_label_statement(i);
  label_of(i);
};

template <class I> consteval bool is_goto(I s) {
  if constexpr (has_jump_api<I>)
    return is_goto_statement(s);
  else
    return false;
}

template <class I> consteval bool is_label(I s) {
  if constexpr (has_jump_api<I>)
    return is_label_statement(s);
  else
    return false;
}

template <class I> consteval I label(I s) {
  if constexpr (has_jump_api<I>)
    return label_of(s);
  else
    throw "smt: this compiler has no label_of";
}

} // namespace detail

// Whether the compiler can reflect gotos and labels, which recognizing a
// lowered loop needs.
inline constexpr bool loops_supported = detail::has_jump_api<m::info>;

namespace detail {

struct term {
  std::string text;
  bool is_bool;
};

struct binding {
  std::string name;
  std::string text;
};

// A pending jump within a loop's body: the label, and the condition under
// which a goto to it has been taken.
struct jump {
  m::info label;
  std::string taken;
};

struct state {
  std::vector<std::pair<std::string, std::string>> env; // C++ name -> symbol
  std::string ret = "(_ bv0 32)";
  std::string returned = "false";
  std::string broken = "false"; // out of the innermost loop by a break
  std::vector<jump> pending;
  std::string path = "true"; // the condition under which this point is reached
};

struct loop_labels {
  m::info exit;
};

// A function's lets, in order, the counter that keeps their names fresh, the
// unrolling bound, and the condition of each loop that its unrolling was
// enough.
struct context {
  std::vector<binding> lets;
  int fresh = 0;
  int unroll = 0;
  std::vector<std::string> unwound;
  std::vector<loop_labels> loops;
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

consteval std::string or_(const std::string &a, const std::string &b) {
  if (a == "true" || b == "true")
    return "true";
  if (a == "false")
    return b;
  if (b == "false" || a == b)
    return a;
  return "(or " + a + " " + b + ")";
}

consteval std::string and_(const std::string &a, const std::string &b) {
  if (a == "false" || b == "false")
    return "false";
  if (a == "true")
    return b;
  if (b == "true" || a == b)
    return a;
  return "(and " + a + " " + b + ")";
}

consteval std::string not_(const std::string &a) {
  if (a == "true")
    return "false";
  if (a == "false")
    return "true";
  return "(not " + a + ")";
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

// `(ite COND A B)`, bound to a fresh name for NAME, unless A and B are the
// same.
consteval std::string ite(context &c, const std::string &name,
                          const std::string &cond, const std::string &a,
                          const std::string &b) {
  if (a == b)
    return a;
  return bind(c, name, "(ite " + cond + " " + a + " " + b + ")");
}

// Where the statements at this point don't run: after a return, a break, or a
// goto past them.
consteval std::string skip(const state &s) {
  std::string r = or_(s.returned, s.broken);
  for (const jump &j : s.pending)
    r = or_(r, j.taken);
  return r;
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

// Where the statement doesn't run, keep OLD; elsewhere take NEW.
consteval std::string guarded(const state &s, const std::string &old_value,
                              const std::string &new_value) {
  const std::string sk = skip(s);
  if (sk == "false")
    return new_value;
  return "(ite " + sk + " " + old_value + " " + new_value + ")";
}

consteval void assign(context &c, state &s, const std::string &name,
                      const term &value) {
  std::string old_value = "(_ bv0 32)";
  for (auto &e : s.env)
    if (e.first == name)
      old_value = e.second;
  set(s, name, bind(c, name, guarded(s, old_value, as_bv(value))));
}

// FLAG becomes true where the statement runs. Where FLAG is all of skip, that
// is everywhere.
consteval std::string taken(context &c, const state &s, const std::string &flag,
                            const char *name) {
  const std::string sk = skip(s);
  if (sk == "false" || sk == flag)
    return "true";
  return bind(c, name, or_(flag, not_(sk)));
}

consteval std::string pending_of(const state &s, m::info l) {
  for (const jump &j : s.pending)
    if (j.label == l)
      return j.taken;
  return "false";
}

consteval void set_pending(state &s, m::info l, std::string cond) {
  for (jump &j : s.pending)
    if (j.label == l) {
      j.taken = std::move(cond);
      return;
    }
  s.pending.push_back({l, std::move(cond)});
}

// The state after an if: A where COND holds, B where it doesn't. Variables
// declared in a branch go out of scope.
consteval state merge(context &c, const std::string &cond, const state &before,
                      const state &a, const state &b) {
  state r = before;
  for (const auto &e : before.env) {
    const std::string x = lookup(a, e.first), y = lookup(b, e.first);
    set(r, e.first, ite(c, e.first, cond, x, y));
  }
  r.ret = ite(c, "ret", cond, a.ret, b.ret);
  r.returned = ite(c, "returned", cond, a.returned, b.returned);
  r.broken = ite(c, "broken", cond, a.broken, b.broken);
  r.pending.clear();
  for (const state *side : {&a, &b})
    for (const jump &j : side->pending)
      if (pending_of(r, j.label) == "false")
        set_pending(r, j.label,
                    ite(c, "jump", cond, pending_of(a, j.label),
                        pending_of(b, j.label)));
  r.path = before.path;
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

// An assignment `x = e`, `x op= e`, `++x` or `x--`, as an expression.
consteval bool assignment(context &c, state &s, m::info e) {
  e = strip(e);
  if (!m::is_operator_expression(e))
    return false;
  const m::operators op = m::expression_operator_of(e);
  const auto ops = m::operands_of(e);
  if ((op == m::operators::op_plus_plus ||
       op == m::operators::op_minus_minus) &&
      ops.size() == 1) {
    const std::string name = name_of(strip(ops[0]));
    assign(c, s, name,
           binary(op == m::operators::op_plus_plus ? m::operators::op_plus
                                                   : m::operators::op_minus,
                  term{lookup(s, name), false}, term{bv(1), false}));
    return true;
  }
  if (!is_assignment(op))
    return false;
  const std::string name = name_of(strip(ops[0]));
  term value = expr(ops[1], s);
  if (op != m::operators::op_equals)
    value = binary(op, term{lookup(s, name), false}, value);
  assign(c, s, name, value);
  return true;
}

// A lowered loop at position J of a block's statements K:
//   while:    [goto C, B:, body..., C:, if (c) goto B; else goto X;, X:]
//   do/while: [B:, body..., if (c) goto B; else goto X;, X:]
// A for's init-statement comes before it, and its increment is the end of the
// body.
struct loop_match {
  bool found = false;
  bool is_do = false;
  std::size_t body_begin = 0, body_end = 0, end = 0;
  m::info cond_if, exit;
};

// Whether K[P] is `if (c) goto B; else goto X;` and K[P + 1] is `X:`.
consteval bool loop_condition_at(const std::vector<m::info> &k, std::size_t p,
                                 m::info b) {
  if (p + 1 >= k.size() || !m::is_if_statement(k[p]) || !is_label(k[p + 1]))
    return false;
  const m::info t = m::true_expression_of(k[p]);
  const m::info f = m::false_expression_of(k[p]);
  return is_goto(t) && is_goto(f) && label(t) == b &&
         label(f) == label(k[p + 1]);
}

consteval loop_match match_loop(const std::vector<m::info> &k, std::size_t j) {
  loop_match r;
  if constexpr (!loops_supported)
    return r;
  const std::size_t n = k.size();
  if (j + 1 < n && is_goto(k[j]) && is_label(k[j + 1])) {
    const m::info cond_label = label(k[j]), body = label(k[j + 1]);
    for (std::size_t p = j + 2; p + 2 < n; ++p)
      if (is_label(k[p]) && label(k[p]) == cond_label &&
          loop_condition_at(k, p + 1, body))
        return {true, false, j + 2, p, p + 3, k[p + 1], label(k[p + 2])};
  }
  if (is_label(k[j])) {
    const m::info body = label(k[j]);
    for (std::size_t p = j + 1; p + 1 < n; ++p)
      if (loop_condition_at(k, p, body))
        return {true, true, j + 1, p, p + 2, k[p], label(k[p + 1])};
  }
  return r;
}

consteval void stmt(context &c, state &s, m::info st);

consteval void block(context &c, state &s, const std::vector<m::info> &k,
                     std::size_t begin, std::size_t end);

consteval void run_loop(context &c, state &s, const std::vector<m::info> &k,
                        const loop_match &lm) {
  if (c.unroll <= 0)
    throw "smt: a loop needs an unrolling bound";
  const m::info cond = m::condition_of(lm.cond_if);
  const std::string outer_broken = s.broken;
  s.broken = "false";
  c.loops.push_back({lm.exit});
  for (int it = 0; it < c.unroll; ++it) {
    const std::string go =
        lm.is_do && it == 0 ? std::string("true") : as_bool(expr(cond, s));
    const std::string enter = and_(not_(skip(s)), go);
    state a = s;
    a.returned = a.broken = "false";
    a.pending.clear();
    a.path = and_(s.path, enter);
    block(c, a, k, lm.body_begin, lm.body_end);
    a.pending.clear(); // the condition is reached again
    s = merge(c, enter, s, a, s);
  }
  // Another iteration would have started nowhere.
  const std::string more =
      and_(and_(s.path, not_(skip(s))), as_bool(expr(cond, s)));
  c.unwound.push_back(not_(more));
  c.loops.pop_back();
  s.broken = outer_broken;
}

consteval void block(context &c, state &s, const std::vector<m::info> &k,
                     std::size_t begin, std::size_t end) {
  for (std::size_t j = begin; j < end;) {
    const loop_match lm = match_loop(k, j);
    if (lm.found && lm.end <= end) {
      run_loop(c, s, k, lm);
      j = lm.end;
    } else
      stmt(c, s, k[j++]);
  }
}

consteval void stmt(context &c, state &s, m::info st) {
  if (st == m::info{})
    return;
  if (m::is_compound_statement(st)) {
    const std::vector<m::info> k = m::statements_of(st);
    block(c, s, k, 0, k.size());
    return;
  }
  if (m::is_return_statement(st)) {
    term v = expr(m::return_operand_of(st), s);
    s.ret = bind(c, "ret", guarded(s, s.ret, as_bv(v)));
    s.returned = taken(c, s, s.returned, "returned");
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
    a.path = and_(s.path, cond);
    stmt(c, a, then_of(st));
    state b = s;
    b.path = and_(s.path, not_(cond));
    stmt(c, b, else_of(st));
    s = merge(c, cond, s, a, b);
    return;
  }
  if (is_goto(st) && !c.loops.empty()) {
    const m::info target = label(st);
    if (target == c.loops.back().exit)
      s.broken = taken(c, s, s.broken, "broken");
    else // a continue: skip up to the label
      set_pending(s, target, taken(c, s, pending_of(s, target), "jump"));
    return;
  }
  if (is_label(st) && !c.loops.empty()) {
    // Where a goto skipped to here, statements run again.
    for (jump &j : s.pending)
      if (j.label == label(st))
        j.taken = "false";
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
// function over int. With loops, UNROLL is how many times to unroll each one,
// and a second `(define-fun NAME_unwound (...) Bool ...)` says whether that
// was enough.
consteval std::string define_fun(m::info fn, int unroll = 0) {
  detail::context c;
  c.unroll = unroll;
  detail::state s;
  std::string params;
  for (m::info p : m::parameters_of(fn)) {
    const std::string name(m::identifier_of(p));
    params += (params.empty() ? "(" : " (") + name + " (_ BitVec 32))";
    s.env.push_back({name, name});
  }
  detail::stmt(c, s, m::body_of(fn));
  // Front to back: wrapping the body in each let in turn would copy it once
  // per let.
  auto with_lets = [&](const std::string &body) {
    std::string r;
    for (const detail::binding &b : c.lets) {
      r += "(let ((";
      r += b.name;
      r += " ";
      r += b.text;
      r += "))\n  ";
    }
    r += body;
    r.append(c.lets.size(), ')');
    return r;
  };
  const std::string name(m::identifier_of(fn));
  std::string r = "(define-fun " + name + " (" + params +
                  ") (_ BitVec 32)\n  " + with_lets(s.ret) + ")\n";
  if (!c.unwound.empty()) {
    std::string all = "true";
    for (const std::string &u : c.unwound)
      all = detail::and_(all, u);
    r += "(define-fun " + name + "_unwound (" + params + ") Bool\n  " +
         with_lets(all) + ")\n";
  }
  return r;
}

} // namespace smt

#endif
