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
// smt::verify(^^f, ^^pre, ^^post) proves f for every input instead, with a
// loop invariant written in the loop as smt::invariant([&] { return ...; }).
// It gives a script of verification conditions, each of which a solver should
// find unsat: the invariant holds when the loop is entered and after each
// iteration, no arithmetic in f overflows, and the result satisfies post. pre
// and post are constexpr bool functions: pre takes f's parameters, post those
// and then the result. There an int is a mathematical integer, and every `+ -
// *` in f is checked to fit in 32 bits, which makes it mean the same as on the
// hardware; pre, post and the invariants are mathematics, with no overflow.
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

// smt::invariant([&] { return cond; }), as a statement of a loop's body, says
// that cond holds whenever the loop's condition is about to be checked. It
// does nothing when the program runs: the lambda is never called, so cond
// isn't evaluated and can't overflow. verify reads cond from the body of the
// lambda's call operator, where a captured variable is the function's own.
template <class F> inline void invariant(F &&) noexcept {}

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

// Something verify's script asks the solver to refute: WHAT for the reader,
// CLAIM, and how many of the hypotheses held where it was made.
struct obligation {
  std::string what;
  std::string claim;
  std::size_t hyps;
};

// A function's lets, in order, the counter that keeps their names fresh, the
// unrolling bound, and the condition of each loop that its unrolling was
// enough. For verify (INTS): the declarations in order, the hypotheses, and
// the obligations; GHOST while translating pre, post or an invariant.
struct context {
  std::vector<binding> lets;
  int fresh = 0;
  int unroll = 0;
  std::vector<std::string> unwound;
  std::vector<loop_labels> loops;
  bool ints = false;
  bool ghost = false;
  std::string decls;
  std::vector<std::string> hyps;
  std::vector<obligation> obligations;
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

consteval std::string num(const context &c, int v) {
  if (!c.ints)
    return bv(v);
  const long long w = v;
  return w < 0 ? "(- " + std::to_string(-w) + ")" : std::to_string(w);
}

consteval std::string zero(const context &c) {
  return c.ints ? "0" : "(_ bv0 32)";
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

consteval std::string implies(const std::string &a, const std::string &b) {
  if (a == "true")
    return b;
  if (a == "false" || b == "true")
    return "true";
  return "(=> " + a + " " + b + ")";
}

consteval std::string as_bool(const context &c, const term &t) {
  return t.is_bool ? t.text : "(not (= " + t.text + " " + num(c, 0) + "))";
}

consteval std::string as_num(const context &c, const term &t) {
  return t.is_bool ? "(ite " + t.text + " " + num(c, 1) + " " + num(c, 0) + ")"
                   : t.text;
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

// Bind TEXT, of sort Bool if IS_BOOL, to a fresh name for NAME and return the
// name: a let for define_fun, a define-fun for verify.
consteval std::string bind(context &c, const std::string &name,
                           std::string text, bool is_bool = false) {
  std::string symbol = name + "!" + std::to_string(++c.fresh);
  if (c.ints)
    c.decls += "(define-fun " + symbol + " () " + (is_bool ? "Bool " : "Int ") +
               text + ")\n";
  else
    c.lets.push_back({symbol, std::move(text)});
  return symbol;
}

// `(ite COND A B)`, bound to a fresh name for NAME, unless A and B are the
// same.
consteval std::string ite(context &c, const std::string &name,
                          const std::string &cond, const std::string &a,
                          const std::string &b, bool is_bool = false) {
  if (a == b)
    return a;
  return bind(c, name, "(ite " + cond + " " + a + " " + b + ")", is_bool);
}

// Where the statements at this point don't run: after a return, a break, or a
// goto past them.
consteval std::string skip(const state &s) {
  std::string r = or_(s.returned, s.broken);
  for (const jump &j : s.pending)
    r = or_(r, j.taken);
  return r;
}

consteval void require(context &c, std::string what, std::string claim) {
  c.obligations.push_back({std::move(what), std::move(claim), c.hyps.size()});
}

// For verify: T, computed where S says, fits in an int.
consteval void check_fits(context &c, const state &s, const std::string &t) {
  if (c.ints && !c.ghost)
    require(c, "no overflow in " + t,
            implies(and_(s.path, not_(skip(s))), "(fits " + t + ")"));
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

// The function that E calls, or a null reflection if E isn't a call.
consteval m::info callee(m::info e) {
  e = strip(e);
  if (!m::is_expression(e) || !m::is_function_call(e))
    return {};
  m::info f = m::callee_of(e);
  return m::is_function(f) ? f : m::declaration_of(f);
}

// The condition of smt::invariant([&] { return cond; }), if E is one: the
// operand of the return statement of the lambda's call operator, found from
// the specialization of smt::invariant that E calls.
consteval m::info invariant_of(m::info e) {
  const m::info f = callee(e);
  if (f == m::info{} || !m::has_template_arguments(f) ||
      m::template_of(f) != ^^smt::invariant)
    return {};
  const m::info closure = m::remove_cvref(m::template_arguments_of(f).front());
  for (m::info mem : m::members_of(closure, m::access_context::unchecked()))
    if (m::is_function(mem) && m::has_body(mem) &&
        !m::is_special_member_function(mem))
      for (m::info s : m::statements_of(m::body_of(mem)))
        if (m::is_return_statement(s))
          return m::return_operand_of(s);
  throw "smt: write an invariant as smt::invariant([&] { return ...; })";
}

consteval term expr(context &c, m::info e, const state &s);

consteval term binary(context &c, const state &s, m::operators op,
                      const term &l, const term &r) {
  using enum m::operators;
  auto arith = [&](const char *bvf, const char *intf) {
    term t{std::string("(") + (c.ints ? intf : bvf) + " " + as_num(c, l) + " " +
               as_num(c, r) + ")",
           false};
    check_fits(c, s, t.text);
    return t;
  };
  auto cmp = [&](const char *bvf, const char *intf) {
    return term{std::string("(") + (c.ints ? intf : bvf) + " " + as_num(c, l) +
                    " " + as_num(c, r) + ")",
                true};
  };
  auto logic = [&](const char *f) {
    return term{std::string("(") + f + " " + as_bool(c, l) + " " +
                    as_bool(c, r) + ")",
                true};
  };
  if (op == op_plus || op == op_plus_equals)
    return arith("bvadd", "+");
  if (op == op_minus || op == op_minus_equals)
    return arith("bvsub", "-");
  if (op == op_star || op == op_star_equals)
    return arith("bvmul", "*");
  if (op == op_less)
    return cmp("bvslt", "<");
  if (op == op_less_equals)
    return cmp("bvsle", "<=");
  if (op == op_greater)
    return cmp("bvsgt", ">");
  if (op == op_greater_equals)
    return cmp("bvsge", ">=");
  if (op == op_equals_equals)
    return cmp("=", "=");
  if (op == op_exclamation_equals)
    return term{"(not (= " + as_num(c, l) + " " + as_num(c, r) + "))", true};
  if (op == op_ampersand_ampersand)
    return logic("and");
  if (op == op_pipe_pipe)
    return logic("or");
  throw "smt: unsupported binary operator";
}

consteval term expr(context &c, m::info e, const state &s) {
  e = strip(e);
  if (m::is_literal(e))
    return {num(c, m::extract<int>(m::constant_of(e))), false};
  if (m::is_id_expression(e))
    return {lookup(s, name_of(e)), false};
  if (m::is_conditional_operator(e)) {
    term k = expr(c, m::condition_of(e), s);
    term t = expr(c, m::true_expression_of(e), s);
    term f = expr(c, m::false_expression_of(e), s);
    if (t.is_bool && f.is_bool)
      return {"(ite " + as_bool(c, k) + " " + t.text + " " + f.text + ")",
              true};
    return {"(ite " + as_bool(c, k) + " " + as_num(c, t) + " " + as_num(c, f) +
                ")",
            false};
  }
  if (m::is_operator_expression(e)) {
    const m::operators op = m::expression_operator_of(e);
    const std::vector<m::info> ops = m::operands_of(e);
    if (ops.size() == 1) {
      term a = expr(c, ops[0], s);
      if (op == m::operators::op_minus) {
        term t{c.ints ? "(- " + as_num(c, a) + ")"
                      : "(bvneg " + as_num(c, a) + ")",
               false};
        check_fits(c, s, t.text);
        return t;
      }
      if (op == m::operators::op_plus)
        return {as_num(c, a), false};
      if (op == m::operators::op_exclamation)
        return {"(not " + as_bool(c, a) + ")", true};
      throw "smt: unsupported unary operator";
    }
    if (ops.size() == 2 && !is_assignment(op))
      return binary(c, s, op, expr(c, ops[0], s), expr(c, ops[1], s));
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
  std::string old_value = zero(c);
  for (auto &e : s.env)
    if (e.first == name)
      old_value = e.second;
  set(s, name, bind(c, name, guarded(s, old_value, as_num(c, value))));
}

// FLAG becomes true where the statement runs. Where FLAG is all of skip, that
// is everywhere.
consteval std::string taken(context &c, const state &s, const std::string &flag,
                            const char *name) {
  const std::string sk = skip(s);
  if (sk == "false" || sk == flag)
    return "true";
  return bind(c, name, or_(flag, not_(sk)), true);
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
  r.returned = ite(c, "returned", cond, a.returned, b.returned, true);
  r.broken = ite(c, "broken", cond, a.broken, b.broken, true);
  r.pending.clear();
  for (const state *side : {&a, &b})
    for (const jump &j : side->pending)
      if (pending_of(r, j.label) == "false")
        set_pending(r, j.label,
                    ite(c, "jump", cond, pending_of(a, j.label),
                        pending_of(b, j.label), true));
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

// The variable that the assignment E (`x = e`, `x op= e`, `++x`, `x--`)
// assigns, or "" if E isn't one.
consteval std::string assigned_by(m::info e) {
  e = strip(e);
  if (!m::is_expression(e) || !m::is_operator_expression(e))
    return "";
  const m::operators op = m::expression_operator_of(e);
  if (op == m::operators::op_plus_plus || op == m::operators::op_minus_minus ||
      is_assignment(op))
    return name_of(strip(m::operands_of(e)[0]));
  return "";
}

// Add to OUT the variables that statement ST assigns.
consteval void assigned_in(m::info st, std::vector<std::string> &out) {
  if (st == m::info{})
    return;
  std::string name;
  if (m::is_compound_statement(st)) {
    for (m::info k : m::statements_of(st))
      assigned_in(k, out);
    return;
  }
  if (m::is_if_statement(st)) {
    assigned_in(then_of(st), out);
    assigned_in(else_of(st), out);
    return;
  }
  if (m::is_expression_statement(st))
    name = assigned_by(m::expression_of(st));
  else if (m::is_expression(st))
    name = assigned_by(st);
  if (!name.empty())
    out.push_back(name);
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
           binary(c, s,
                  op == m::operators::op_plus_plus ? m::operators::op_plus
                                                   : m::operators::op_minus,
                  term{lookup(s, name), false}, term{num(c, 1), false}));
    return true;
  }
  if (!is_assignment(op))
    return false;
  const std::string name = name_of(strip(ops[0]));
  term value = expr(c, ops[1], s);
  if (op != m::operators::op_equals)
    value = binary(c, s, op, term{lookup(s, name), false}, value);
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
    const std::string go = lm.is_do && it == 0 ? std::string("true")
                                               : as_bool(c, expr(c, cond, s));
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
      and_(and_(s.path, not_(skip(s))), as_bool(c, expr(c, cond, s)));
  c.unwound.push_back(not_(more));
  c.loops.pop_back();
  s.broken = outer_broken;
}

// Set INV to the invariant among statement ST and the blocks in it, but not
// in a nested loop's body or an if's branches. The lambda's closure is a
// temporary, which makes a loop's body a block of its own.
consteval void find_invariant(m::info st, m::info &inv) {
  if (m::is_compound_statement(st)) {
    for (m::info k : m::statements_of(st))
      find_invariant(k, inv);
    return;
  }
  if (!m::is_expression_statement(st))
    return;
  if (m::info i = invariant_of(m::expression_of(st)); i != m::info{}) {
    if (inv != m::info{})
      throw "smt: a loop has more than one smt::invariant";
    inv = i;
  }
}

// The invariant of a loop: the smt::invariant among the statements of its
// body.
consteval m::info loop_invariant(const std::vector<m::info> &k,
                                 const loop_match &lm) {
  m::info inv;
  for (std::size_t j = lm.body_begin; j < lm.body_end; ++j)
    find_invariant(k[j], inv);
  if (inv == m::info{})
    throw "smt: verify needs an smt::invariant(...) in each loop";
  return inv;
}

consteval std::string ghost(context &c, m::info e, const state &s) {
  c.ghost = true;
  const std::string r = as_bool(c, expr(c, e, s));
  c.ghost = false;
  return r;
}

consteval void hypothesis(context &c, const std::string &h) {
  c.decls += "(define-fun hyp!" + std::to_string(c.hyps.size()) + " () Bool " +
             h + ")\n";
  c.hyps.push_back(h);
}

// For verify: replace a loop by its invariant. The invariant must hold on
// entry; then the variables the loop assigns become arbitrary values for which
// it holds, one iteration from there must restore it, and the code after the
// loop starts from where the condition is false or the body broke out.
consteval void verify_loop(context &c, state &s, const std::vector<m::info> &k,
                           const loop_match &lm) {
  if (s.path != "true" || skip(s) != "false")
    throw "smt: verify handles a loop only where every path reaches it";
  const m::info inv = loop_invariant(k, lm);
  const m::info cond = m::condition_of(lm.cond_if);
  require(c, "the loop invariant holds on entry", ghost(c, inv, s));

  std::vector<std::string> names;
  for (std::size_t j = lm.body_begin; j < lm.body_end; ++j)
    assigned_in(k[j], names);
  // Each variable in scope that the loop assigns, once.
  const auto in_scope = s.env;
  for (const auto &e : in_scope)
    for (const std::string &n : names)
      if (n == e.first) {
        const std::string fresh = n + "!" + std::to_string(++c.fresh);
        c.decls += "(declare-const " + fresh + " Int)\n";
        hypothesis(c, "(fits " + fresh + ")");
        set(s, n, fresh);
        break;
      }
  hypothesis(c, ghost(c, inv, s));

  const state head = s;
  const std::string go =
      lm.is_do ? std::string("true") : as_bool(c, expr(c, cond, head));
  state a = head;
  a.path = go;
  c.loops.push_back({lm.exit});
  block(c, a, k, lm.body_begin, lm.body_end);
  c.loops.pop_back();
  a.pending.clear();
  const std::string done = or_(a.returned, a.broken);
  if (lm.is_do) {
    const std::string again = as_bool(c, expr(c, cond, a));
    require(c, "the loop invariant is preserved",
            implies(and_(not_(done), again), ghost(c, inv, a)));
    s = a;
    hypothesis(c, or_(done, not_(again)));
  } else {
    require(c, "the loop invariant is preserved",
            implies(and_(go, not_(done)), ghost(c, inv, a)));
    s = merge(c, go, head, a, head);
    hypothesis(c, or_(not_(go), done));
  }
  s.broken = "false";
  s.path = "true";
}

consteval void block(context &c, state &s, const std::vector<m::info> &k,
                     std::size_t begin, std::size_t end) {
  for (std::size_t j = begin; j < end;) {
    const loop_match lm = match_loop(k, j);
    if (lm.found && lm.end <= end) {
      if (c.ints)
        verify_loop(c, s, k, lm);
      else
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
    term v = expr(c, m::return_operand_of(st), s);
    s.ret = bind(c, "ret", guarded(s, s.ret, as_num(c, v)));
    s.returned = taken(c, s, s.returned, "returned");
    return;
  }
  if (m::is_declaration_statement(st)) {
    for (m::info var : m::declarators_of(st)) {
      m::info init = m::initializer_of(var);
      if (init == m::info{})
        throw "smt: a declaration needs an initializer";
      assign(c, s, std::string(m::identifier_of(var)), expr(c, init, s));
    }
    return;
  }
  if (m::is_expression_statement(st)) {
    const m::info e = m::expression_of(st);
    if (invariant_of(e) != m::info{}) // read by verify_loop
      return;
    if (!assignment(c, s, e))
      throw "smt: unsupported expression statement";
    return;
  }
  if (m::is_if_statement(st)) {
    const std::string cond = as_bool(c, expr(c, m::condition_of(st), s));
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

// What the constexpr bool function SPEC returns for ARGS, as mathematics.
consteval std::string spec(context &c, m::info fn,
                           const std::vector<std::string> &args) {
  const auto params = m::parameters_of(fn);
  if (params.size() != args.size())
    throw "smt: a pre or post has the wrong number of parameters";
  state s;
  for (std::size_t i = 0; i < args.size(); ++i)
    s.env.push_back({std::string(m::identifier_of(params[i])), args[i]});
  const auto body = m::statements_of(m::body_of(fn));
  if (body.size() != 1 || !m::is_return_statement(body[0]))
    throw "smt: a pre or post must be a single return statement";
  return ghost(c, m::return_operand_of(body[0]), s);
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

// The verification conditions of FN against PRE and POST, as an SMT-LIB
// script: each `(check-sat)` after an `(echo ...)` saying what it checks
// should be unsat.
consteval std::string verify(m::info fn, m::info pre, m::info post) {
  detail::context c;
  c.ints = true;
  detail::state s;
  s.ret = "0";
  std::vector<std::string> args;
  for (m::info p : m::parameters_of(fn)) {
    const std::string name(m::identifier_of(p));
    c.decls += "(declare-const " + name + " Int)\n";
    detail::hypothesis(c, "(fits " + name + ")");
    s.env.push_back({name, name});
    args.push_back(name);
  }
  detail::hypothesis(c, detail::spec(c, pre, args));
  detail::stmt(c, s, m::body_of(fn));
  args.push_back(s.ret);
  detail::require(c, "the function returns, and its result satisfies post",
                  detail::and_(s.returned, detail::spec(c, post, args)));

  const std::string name(m::identifier_of(fn));
  std::string r = "; Verification conditions for " + name +
                  ": every check-sat should be unsat.\n"
                  "(set-logic QF_NIA)\n"
                  "(define-fun fits ((x Int)) Bool\n"
                  "  (and (<= (- 2147483648) x) (<= x 2147483647)))\n" +
                  c.decls;
  for (const detail::obligation &o : c.obligations) {
    r += "(echo \"" + name + ": " + o.what + "\")\n(push)\n";
    for (std::size_t i = 0; i < o.hyps; ++i)
      r += "(assert hyp!" + std::to_string(i) + ")\n";
    r += "(assert (not " + o.claim + "))\n(check-sat)\n(pop)\n";
  }
  return r;
}

} // namespace smt

#endif
