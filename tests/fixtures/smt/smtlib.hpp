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
// A pointer parameter is an array, `a[i]` a read of it, checked to be within
// the array (whose length is unknown unless pre bounds it with
// smt::valid_range), and smt::forall(lo, hi, [&](int k) { return ...; }) in a
// specification or an invariant a quantifier over lo <= k < hi.
//
// See docs/smt-verification.md for how to check the output with a solver.

#ifndef SMT_SMTLIB_HPP
#define SMT_SMTLIB_HPP

#include <ad/meta.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace smt {

// smt::invariant([&] { return cond; }), as a statement of a loop's body, says
// that cond holds whenever the loop's condition is about to be checked. It
// does nothing when the program runs: the lambda is never called, so cond
// isn't evaluated and can't overflow. verify reads cond from the body of the
// lambda's call operator, where a captured variable is the function's own.
template <class F> inline void invariant(F &&) noexcept {}

// For specifications and invariants: whether f(k) holds for every k with
// lo <= k < hi. It can also run, which tests a specification on an input.
template <class F> constexpr bool forall(int lo, int hi, F f) {
  for (int k = lo; k < hi; ++k)
    if (!f(k))
      return false;
  return true;
}

// For specifications: a points to at least n ints. Running, it can't tell,
// so it says yes.
constexpr bool valid_range(const int *, int) { return true; }

// For invariants: the value x had when the function was entered, such as
// smt::old(a)[k] for an array. Running, it gives x as it is now.
template <class T> constexpr T old(T x) { return x; }

// For specifications and invariants: a[lo, hi) holds the same values as
// b[lo, hi), each as many times. It can also run.
constexpr bool permutation(const int *a, const int *b, int lo, int hi) {
  for (int k = lo; k < hi; ++k) {
    int in_a = 0, in_b = 0;
    for (int j = lo; j < hi; ++j) {
      in_a += a[j] == a[k];
      in_b += b[j] == a[k];
    }
    if (in_a != in_b)
      return false;
  }
  return true;
}

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
  std::vector<std::pair<std::string, std::string>> arrays; // symbol -> length
  bool quantified = false;
  bool permutations = false; // the script needs perm and its axioms
  state entry;               // the parameters, for smt::old
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

// Bind TEXT, of sort SORT, to a fresh name for NAME and return the name: a
// let for define_fun, a define-fun for verify.
consteval std::string bind(context &c, const std::string &name,
                           std::string text, const char *sort) {
  std::string symbol = name + "!" + std::to_string(++c.fresh);
  if (c.ints)
    c.decls += "(define-fun " + symbol + " () " + sort + " " + text + ")\n";
  else
    c.lets.push_back({symbol, std::move(text)});
  return symbol;
}

consteval std::string bind(context &c, const std::string &name,
                           std::string text, bool is_bool = false) {
  return bind(c, name, std::move(text), is_bool ? "Bool" : "Int");
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

// Whether F, a function, is a specialization of the function template TMPL.
consteval bool specializes(m::info f, m::info tmpl) {
  return f != m::info{} && m::has_template_arguments(f) &&
         m::template_of(f) == tmpl;
}

// The call operator of the lambda whose closure type is the first template
// argument of F, a specialization such as smt::invariant<closure>.
consteval m::info lambda_operator(m::info f) {
  const m::info closure = m::remove_cvref(m::template_arguments_of(f).front());
  for (m::info mem : m::members_of(closure, m::access_context::unchecked()))
    if (m::is_function(mem) && m::has_body(mem) &&
        !m::is_special_member_function(mem))
      return mem;
  throw "smt: expected a lambda";
}

// The operand of the return statement in the block BODY, or a null
// reflection.
consteval m::info return_in(m::info body) {
  for (m::info s : m::statements_of(body)) {
    if (m::is_return_statement(s))
      return m::return_operand_of(s);
    if (m::is_compound_statement(s))
      if (m::info r = return_in(s); r != m::info{})
        return r;
  }
  return {};
}

// The operand of the return statement of the function OP.
consteval m::info returned_by(m::info op) {
  if (m::info r = return_in(m::body_of(op)); r != m::info{})
    return r;
  throw "smt: a lambda in a specification must return its condition";
}

// The condition of smt::invariant([&] { return cond; }), if E is one: the
// operand of the return statement of the lambda's call operator, found from
// the specialization of smt::invariant that E calls.
consteval m::info invariant_of(m::info e) {
  const m::info f = callee(e);
  if (!specializes(f, ^^smt::invariant))
    return {};
  return returned_by(lambda_operator(f));
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
  if (op == op_slash || op == op_slash_equals) {
    if (!c.ints)
      return arith("bvsdiv", "div");
    // C++ rounds toward zero; SMT-LIB's div rounds so that the remainder
    // isn't negative. They agree on non-negative operands.
    const std::string x = as_num(c, l), y = as_num(c, r);
    // A positive constant divisor can neither fail nor overflow, and only
    // the sign of x matters.
    const bool safe = !y.empty() && y != "0" &&
                      y.find_first_not_of("0123456789") == std::string::npos;
    const std::string ax = "(ite (>= " + x + " 0) " + x + " (- " + x + "))";
    const std::string ay =
        safe ? y : "(ite (>= " + y + " 0) " + y + " (- " + y + "))";
    const std::string q = "(div " + ax + " " + ay + ")";
    const std::string same_sign =
        safe ? "(>= " + x + " 0)" : "(= (>= " + x + " 0) (> " + y + " 0))";
    term t{"(ite " + same_sign + " " + q + " (- " + q + "))", false};
    if (!c.ghost && !safe) {
      require(c, "no division by zero in (/ " + x + " " + y + ")",
              implies(and_(s.path, not_(skip(s))), "(not (= " + y + " 0))"));
      check_fits(c, s, t.text);
    }
    return t;
  }
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

// The length symbol of the array SYMBOL.
consteval std::string length_of(const context &c, const std::string &symbol) {
  for (const auto &a : c.arrays)
    if (a.first == symbol)
      return a.second;
  throw "smt: subscript of something that isn't an array parameter";
}

consteval bool is_array(const context &c, const std::string &symbol) {
  for (const auto &a : c.arrays)
    if (a.first == symbol)
      return true;
  return false;
}

// Bind TEXT, an array with the length of the array LIKE, to a fresh name for
// NAME.
consteval std::string bind_array(context &c, const std::string &name,
                                 std::string text, const std::string &like) {
  const std::string len = length_of(c, like);
  const std::string symbol = bind(c, name, std::move(text), "(Array Int Int)");
  c.arrays.push_back({symbol, len});
  return symbol;
}

// The array that E names: an array parameter, or smt::old of one.
consteval std::string array_of(const context &c, m::info e, const state &s) {
  e = strip(e);
  if (specializes(callee(e), ^^smt::old))
    return array_of(c, m::operands_of(e).front(), c.entry);
  if (!m::is_id_expression(e))
    throw "smt: only an array parameter can be subscripted";
  const std::string symbol = lookup(s, name_of(e));
  length_of(c, symbol);
  return symbol;
}

consteval term expr(context &c, m::info e, const state &s);

// An element of an array: the C++ name of the array ("" for smt::old of
// one), its symbol, and the index.
struct element {
  std::string name, array, index;
};

// The element that E, a subscript `p[i]` or a dereference `*p`, denotes. The
// pointer P can be an array, smt::old of one, or such a pointer plus an
// offset: GCC folds `a[j - 1]` to `*(a + (j - 1))`, and a constant term of an
// index can end up in the pointer.
consteval element element_of(context &c, m::info e, const state &s) {
  e = strip(e);
  const auto ops = m::operands_of(e);
  std::string index = ops.size() > 1 ? as_num(c, expr(c, ops[1], s)) : "0";
  m::info base = strip(ops[0]);
  while (m::is_operator_expression(base) &&
         m::expression_operator_of(base) == m::operators::op_plus &&
         m::operands_of(base).size() == 2) {
    const auto parts = m::operands_of(base);
    const std::string k = as_num(c, expr(c, parts[1], s));
    index = index == "0" ? k : "(+ " + k + " " + index + ")";
    base = strip(parts[0]);
  }
  const std::string name =
      m::is_id_expression(base) ? name_of(base) : std::string();
  return {name, array_of(c, base, s), index};
}

// For verify: index I of the array A, read or written where S says, is within
// it.
consteval void check_bounds(context &c, const state &s, const std::string &a,
                            const std::string &i) {
  if (!c.ghost)
    require(c, "index in bounds: " + a + "[" + i + "]",
            implies(and_(s.path, not_(skip(s))), "(and (<= 0 " + i + ") (< " +
                                                     i + " " + length_of(c, a) +
                                                     "))"));
}

consteval term expr(context &c, m::info e, const state &s) {
  e = strip(e);
  if (m::is_literal(e)) {
    const m::info v = m::constant_of(e);
    const m::info t = m::type_of(v);
    if (t == ^^bool)
      return {m::extract<bool>(v) ? "true" : "false", true};
    // An index that the compiler computed, as for a[1], is a ptrdiff_t, and
    // a constant in a folded index a size_t, which wraps: j - 1 can be
    // j + 0xffffffffffffffff. Read those as the signed values they mean.
    long long value;
    if (t == ^^int)
      value = m::extract<int>(v);
    else if (t == ^^long)
      value = m::extract<long>(v);
    else if (t == ^^unsigned long)
      value = static_cast<long long>(m::extract<unsigned long>(v));
    else if (t == ^^long long)
      value = m::extract<long long>(v);
    else if (t == ^^unsigned long long)
      value = static_cast<long long>(m::extract<unsigned long long>(v));
    else if (t == ^^unsigned)
      value = m::extract<unsigned>(v);
    else
      throw "smt: unsupported literal type";
    if (value < -2147483648LL || value > 2147483647LL)
      throw "smt: a literal out of the range of int";
    return {num(c, static_cast<int>(value)), false};
  }
  if (m::is_id_expression(e)) {
    const std::string symbol = lookup(s, name_of(e));
    for (const auto &a : c.arrays)
      if (a.first == symbol)
        throw "smt: an array is only supported subscripted";
    return {symbol, false};
  }
  if (const m::info f = callee(e); f != m::info{}) {
    const std::vector<m::info> args = m::operands_of(strip(e));
    if (specializes(f, ^^smt::forall)) {
      if (!c.ghost)
        throw "smt: smt::forall is for specifications and invariants";
      const std::string lo = as_num(c, expr(c, args[0], s));
      const std::string hi = as_num(c, expr(c, args[1], s));
      const m::info op = lambda_operator(f);
      const std::string k(m::identifier_of(m::parameters_of(op).front()));
      const std::string bound = k + "!" + std::to_string(++c.fresh);
      state inner = s;
      inner.env.push_back({k, bound});
      const std::string body = as_bool(c, expr(c, returned_by(op), inner));
      c.quantified = true;
      return {"(forall ((" + bound + " Int)) (=> (and (<= " + lo + " " + bound +
                  ") (< " + bound + " " + hi + ")) " + body + "))",
              true};
    }
    if (f == ^^smt::valid_range)
      return {"(>= " + length_of(c, array_of(c, args[0], s)) + " " +
                  as_num(c, expr(c, args[1], s)) + ")",
              true};
    if (!c.ghost) {
      if (!c.ints && m::is_function(f)) {
        std::string call = "(" + std::string(m::identifier_of(f));
        for (m::info arg : args)
          call += " " + as_num(c, expr(c, arg, s));
        call += ")";
        return {call, false};
      }
      throw "smt: unsupported call";
    }
    if (specializes(f, ^^smt::old))
      return expr(c, args[0], c.entry);
    if (f == ^^smt::permutation) {
      c.permutations = true;
      return {"(perm " + array_of(c, args[0], s) + " " +
                  array_of(c, args[1], s) + " " +
                  as_num(c, expr(c, args[2], s)) + " " +
                  as_num(c, expr(c, args[3], s)) + ")",
              true};
    }
    // Another function of a specification: inline its return statement.
    if (m::is_function(f) && m::has_body(f)) {
      const auto params = m::parameters_of(f);
      if (params.size() != args.size())
        throw "smt: unsupported call";
      state inner;
      for (std::size_t i = 0; i < args.size(); ++i)
        inner.env.push_back(
            {std::string(m::identifier_of(params[i])),
             m::is_pointer_type(m::remove_cvref(m::type_of(params[i])))
                 ? array_of(c, args[i], s)
                 : as_num(c, expr(c, args[i], s))});
      return expr(c, returned_by(f), inner);
    }
    throw "smt: unsupported call";
  }
  if (m::is_conditional_operator(e)) {
    term k = expr(c, m::condition_of(e), s);
    state st = s, sf = s; // each branch is evaluated only where it is taken
    st.path = and_(s.path, as_bool(c, k));
    sf.path = and_(s.path, not_(as_bool(c, k)));
    term t = expr(c, m::true_expression_of(e), st);
    term f = expr(c, m::false_expression_of(e), sf);
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
    if ((ops.size() == 1 && op == m::operators::op_star) ||
        (ops.size() == 2 && op == m::operators::op_square_brackets)) {
      if (!c.ints)
        throw "smt: arrays are supported by verify only";
      const element el = element_of(c, e, s);
      check_bounds(c, s, el.array, el.index);
      return {"(select " + el.array + " " + el.index + ")", false};
    }
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
    // The right operand of && and || is evaluated only where the left one
    // doesn't decide.
    if (ops.size() == 2 && (op == m::operators::op_ampersand_ampersand ||
                            op == m::operators::op_pipe_pipe)) {
      const term l = expr(c, ops[0], s);
      state rs = s;
      rs.path =
          and_(s.path, op == m::operators::op_pipe_pipe ? not_(as_bool(c, l))
                                                        : as_bool(c, l));
      return binary(c, s, op, l, expr(c, ops[1], rs));
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
    if (x != y && is_array(c, x))
      set(r, e.first,
          bind_array(c, e.first, "(ite " + cond + " " + x + " " + y + ")", x));
    else
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

consteval bool is_subscript(m::info e);

// The variable that the assignment E (`x = e`, `x op= e`, `++x`, `x--`)
// assigns, or "" if E isn't one.
consteval std::string assigned_by(m::info e) {
  e = strip(e);
  if (!m::is_expression(e) || !m::is_operator_expression(e))
    return "";
  const m::operators op = m::expression_operator_of(e);
  if (op == m::operators::op_plus_plus || op == m::operators::op_minus_minus ||
      is_assignment(op)) {
    m::info target = strip(m::operands_of(e)[0]);
    if (is_subscript(target)) { // a[i] = ... assigns a
      target = strip(m::operands_of(target)[0]);
      while (m::is_operator_expression(target) &&
             m::expression_operator_of(target) == m::operators::op_plus)
        target = strip(m::operands_of(target)[0]);
    }
    return name_of(target);
  }
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

// Whether E is an element of an array: a subscript, `a[i]`, or `*a`, which is
// also what GCC folds `a[0]` to.
consteval bool is_subscript(m::info e) {
  e = strip(e);
  if (!m::is_expression(e) || !m::is_operator_expression(e))
    return false;
  const m::operators op = m::expression_operator_of(e);
  return op == m::operators::op_square_brackets ||
         (op == m::operators::op_star && m::operands_of(e).size() == 1);
}

// An assignment to an element, `a[i] = e`, `a[i] op= e`, `++a[i]` or
// `a[i]--`: the array becomes a store.
consteval void element_assignment(context &c, state &s, m::operators op,
                                  m::info target, m::info rhs) {
  if (!c.ints)
    throw "smt: arrays are supported by verify only";
  const element el = element_of(c, target, s);
  if (el.name.empty())
    throw "smt: only an array parameter can be written";
  const std::string &name = el.name, &a = el.array, &i = el.index;
  check_bounds(c, s, a, i);
  const term old_element{"(select " + a + " " + i + ")", false};
  term value;
  if (op == m::operators::op_plus_plus || op == m::operators::op_minus_minus)
    value = binary(c, s,
                   op == m::operators::op_plus_plus ? m::operators::op_plus
                                                    : m::operators::op_minus,
                   old_element, term{num(c, 1), false});
  else {
    value = expr(c, rhs, s);
    if (op != m::operators::op_equals)
      value = binary(c, s, op, old_element, value);
  }
  set(s, name,
      bind_array(
          c, name,
          guarded(s, a, "(store " + a + " " + i + " " + as_num(c, value) + ")"),
          a));
}

// An assignment `x = e`, `x op= e`, `++x` or `x--`, as an expression.
consteval bool assignment(context &c, state &s, m::info e) {
  e = strip(e);
  if (!m::is_operator_expression(e))
    return false;
  const m::operators op = m::expression_operator_of(e);
  const auto ops = m::operands_of(e);
  if ((op == m::operators::op_plus_plus || op == m::operators::op_minus_minus ||
       is_assignment(op)) &&
      is_subscript(ops[0])) {
    element_assignment(c, s, op, ops[0], ops.size() > 1 ? ops[1] : m::info{});
    return true;
  }
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

consteval void find_invariant(const std::vector<m::info> &k, std::size_t begin,
                              std::size_t end, m::info &inv);

// Set INV to the invariant among statement ST and the blocks in it, but not
// in a nested loop or an if's branches. The lambda's closure is a temporary,
// which makes a loop's body a block of its own.
consteval void find_invariant(m::info st, m::info &inv) {
  if (m::is_compound_statement(st)) {
    const std::vector<m::info> k = m::statements_of(st);
    find_invariant(k, 0, k.size(), inv);
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

// The same among the statements K[BEGIN, END), skipping the loops in them.
consteval void find_invariant(const std::vector<m::info> &k, std::size_t begin,
                              std::size_t end, m::info &inv) {
  for (std::size_t j = begin; j < end;) {
    const loop_match lm = match_loop(k, j);
    if (lm.found && lm.end <= end)
      j = lm.end;
    else
      find_invariant(k[j++], inv);
  }
}

// The invariant of a loop: the smt::invariant among the statements of its
// body.
consteval m::info loop_invariant(const std::vector<m::info> &k,
                                 const loop_match &lm) {
  m::info inv;
  find_invariant(k, lm.body_begin, lm.body_end, inv);
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
// loop starts from where the condition is false or the body broke out. All of
// that is where the loop is reached (ENTER): a loop in an if's branch or in
// another loop's body is reached only on some paths, and elsewhere the state
// stays as it was.
consteval void verify_loop(context &c, state &s, const std::vector<m::info> &k,
                           const loop_match &lm) {
  const std::string enter = and_(s.path, not_(skip(s)));
  const state before = s;
  const m::info inv = loop_invariant(k, lm);
  const m::info cond = m::condition_of(lm.cond_if);
  require(c, "the loop invariant holds on entry",
          implies(enter, ghost(c, inv, s)));

  std::vector<std::string> names;
  for (std::size_t j = lm.body_begin; j < lm.body_end; ++j)
    assigned_in(k[j], names);
  // Each variable in scope that the loop assigns, once.
  const auto in_scope = s.env;
  for (const auto &e : in_scope)
    for (const std::string &n : names)
      if (n == e.first) {
        const std::string fresh = n + "!" + std::to_string(++c.fresh);
        if (is_array(c, e.second)) {
          c.decls += "(declare-const " + fresh + " (Array Int Int))\n";
          c.arrays.push_back({fresh, length_of(c, e.second)});
        } else {
          c.decls += "(declare-const " + fresh + " Int)\n";
          hypothesis(c, "(fits " + fresh + ")");
        }
        set(s, n, fresh);
        break;
      }
  hypothesis(c, implies(enter, ghost(c, inv, s)));

  const state head = s;
  const std::string go =
      lm.is_do ? std::string("true") : as_bool(c, expr(c, cond, head));
  state a = head;
  // Where the loop is entered, nothing has returned, broken out or jumped.
  a.returned = a.broken = "false";
  a.pending.clear();
  a.path = and_(enter, go);
  c.loops.push_back({lm.exit});
  block(c, a, k, lm.body_begin, lm.body_end);
  c.loops.pop_back();
  a.pending.clear();
  const std::string done = or_(a.returned, a.broken);
  state after;
  if (lm.is_do) {
    const std::string again = as_bool(c, expr(c, cond, a));
    require(c, "the loop invariant is preserved",
            implies(and_(enter, and_(not_(done), again)), ghost(c, inv, a)));
    after = a;
    hypothesis(c, implies(enter, or_(done, not_(again))));
  } else {
    require(c, "the loop invariant is preserved",
            implies(and_(enter, and_(go, not_(done))), ghost(c, inv, a)));
    after = merge(c, go, head, a, head);
    hypothesis(c, implies(enter, or_(not_(go), done)));
  }
  after.broken = before.broken;
  after.pending = before.pending;
  s = merge(c, enter, before, after, before);
  s.broken = before.broken;
  s.pending = before.pending;
  s.path = before.path;
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
    m::info operand;
    try {
      operand = m::return_operand_of(st);
    } catch (const std::meta::exception &) { // `return;`
    }
    if (operand != m::info{}) {
      term v = expr(c, operand, s);
      s.ret = bind(c, "ret", guarded(s, s.ret, as_num(c, v)));
    }
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

// perm(a, b, lo, hi): a[lo, hi) is a permutation of b[lo, hi). It is left
// undefined, with only axioms that every permutation satisfies, so whatever
// follows from them holds for permutations: an array is one of itself, and
// swapping two elements in the range keeps one. The swap is written out as
// the stores of `t = a[i]; a[i] = a[j]; a[j] = t;`, which is what the solver
// sees when the code swaps.
inline constexpr const char *permutation_axioms =
    "(declare-fun perm ((Array Int Int) (Array Int Int) Int Int) Bool)\n"
    "(assert (forall ((a (Array Int Int)) (l Int) (h Int))\n"
    "  (! (perm a a l h) :pattern ((perm a a l h)))))\n"
    "(assert (forall ((a (Array Int Int)) (b (Array Int Int)) (l Int) (h Int)\n"
    "                 (i Int) (j Int))\n"
    "  (! (=> (and (perm a b l h) (<= l i) (< i h) (<= l j) (< j h))\n"
    "         (perm (store (store a i (select a j)) j (select a i)) b l h))\n"
    "     :pattern ((perm a b l h)\n"
    "               (store (store a i (select a j)) j (select a i))))))\n";

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

// Generate an SMT-LIB script in QF_BV to check whether CLAIM_FN always returns
// non-zero (1). DEFS are functions that CLAIM_FN calls.
// The script asserts that CLAIM_FN returns 0 / false: `unsat` proves CLAIM_FN
// holds for all inputs, while `sat` finds a counterexample.
consteval std::string refute_script(const std::vector<m::info> &defs,
                                    m::info claim_fn, bool get_model = false) {
  std::string s = "(set-logic QF_BV)\n";
  for (m::info d : defs)
    s += define_fun(d);
  s += define_fun(claim_fn);
  std::string call = "(" + std::string(m::identifier_of(claim_fn));
  std::string get_vals;
  for (m::info p : m::parameters_of(claim_fn)) {
    const std::string name(m::identifier_of(p));
    s += "(declare-const " + name + " (_ BitVec 32))\n";
    call += " " + name;
    get_vals += (get_vals.empty() ? "(" : " ") + name;
  }
  call += ")";
  if (!get_vals.empty())
    get_vals += ")";
  s += "(assert (not (= " + call + " #x00000001)))\n";
  s += "(check-sat)\n";
  if (get_model && !get_vals.empty())
    s += "(get-value " + get_vals + ")\n";
  return s;
}

consteval std::string refute_script(m::info claim_fn, bool get_model = false) {
  return refute_script(std::vector<m::info>{}, claim_fn, get_model);
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
    if (m::is_pointer_type(m::remove_cvref(m::type_of(p)))) {
      // An array, of a length that only pre can tell.
      c.decls += "(declare-const " + name + " (Array Int Int))\n";
      c.decls += "(declare-const " + name + "!len Int)\n";
      detail::hypothesis(c, "(>= " + name + "!len 0)");
      c.arrays.push_back({name, name + "!len"});
    } else {
      c.decls += "(declare-const " + name + " Int)\n";
      detail::hypothesis(c, "(fits " + name + ")");
    }
    s.env.push_back({name, name});
    args.push_back(name);
  }
  c.entry = s;
  detail::hypothesis(c, detail::spec(c, pre, args));
  detail::stmt(c, s, m::body_of(fn));
  // post takes the parameters as they were on entry, then the result, then
  // each array the function can write, as it is at the end.
  const bool returns_value = m::return_type_of(fn) != ^^void;
  if (returns_value)
    args.push_back(s.ret);
  for (m::info p : m::parameters_of(fn)) {
    const m::info t = m::remove_cvref(m::type_of(p));
    if (m::is_pointer_type(t) && !m::is_const_type(m::remove_pointer(t)))
      args.push_back(detail::lookup(s, std::string(m::identifier_of(p))));
  }
  if (returns_value)
    detail::require(c, "the function returns, and its result satisfies post",
                    detail::and_(s.returned, detail::spec(c, post, args)));
  else
    detail::require(c, "post holds when the function returns",
                    detail::spec(c, post, args));

  const std::string name(m::identifier_of(fn));
  std::string r = "; Verification conditions for " + name +
                  ": every check-sat should be unsat.\n" +
                  (c.quantified || !c.arrays.empty() ? "(set-logic ALL)\n"
                                                     : "(set-logic QF_NIA)\n") +
                  "(define-fun fits ((x Int)) Bool\n"
                  "  (and (<= (- 2147483648) x) (<= x 2147483647)))\n" +
                  (c.permutations ? detail::permutation_axioms : "") + c.decls;
  for (const detail::obligation &o : c.obligations) {
    r += "(echo \"" + name + ": " + o.what + "\")\n(push)\n";
    for (std::size_t i = 0; i < o.hyps; ++i)
      r += "(assert hyp!" + std::to_string(i) + ")\n";
    r += "(assert (not " + o.claim + "))\n(check-sat)\n(pop)\n";
  }
  return r;
}

// Run `z3 -in` on SCRIPT and return its stdout, without a trailing newline.
inline std::string z3(std::string_view script) {
  int in_pipe[2], out_pipe[2];
  if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0)
    throw std::runtime_error("smt: pipe failed");
  const pid_t pid = fork();
  if (pid < 0)
    throw std::runtime_error("smt: fork failed");
  if (pid == 0) {
    close(in_pipe[1]);
    close(out_pipe[0]);
    dup2(in_pipe[0], STDIN_FILENO);
    dup2(out_pipe[1], STDOUT_FILENO);
    close(in_pipe[0]);
    close(out_pipe[1]);
    execlp("z3", "z3", "-in", static_cast<char *>(nullptr));
    _exit(127);
  }
  close(in_pipe[0]);
  close(out_pipe[1]);
  const char *p = script.data();
  std::size_t rem = script.size();
  while (rem > 0) {
    const ssize_t n = write(in_pipe[1], p, rem);
    if (n <= 0)
      break;
    p += n;
    rem -= static_cast<std::size_t>(n);
  }
  close(in_pipe[1]);
  std::string out;
  char buf[4096];
  for (;;) {
    const ssize_t n = read(out_pipe[0], buf, sizeof buf);
    if (n <= 0)
      break;
    out.append(buf, static_cast<std::size_t>(n));
  }
  close(out_pipe[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("smt: z3 failed: " + out);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
    out.pop_back();
  return out;
}

// One `(echo "<fn>: <what>")` / `(check-sat)` pair in the output of a script
// from smt::verify.
struct check_result {
  std::string what;
  std::string status;
};

inline std::vector<check_result> z3_checks(std::string_view script) {
  const std::string out = z3(script);
  std::vector<std::string> lines;
  for (std::size_t i = 0; i < out.size();) {
    std::size_t e = out.find('\n', i);
    if (e == std::string::npos)
      e = out.size();
    std::string line = out.substr(i, e - i);
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    lines.push_back(std::move(line));
    i = e + 1;
  }
  if (lines.empty() || lines.size() % 2 != 0)
    throw std::runtime_error("smt: unexpected z3 output:\n" + out);
  std::vector<check_result> r;
  for (std::size_t i = 0; i < lines.size(); i += 2) {
    std::string what = lines[i];
    if (const std::size_t colon = what.find(": "); colon != std::string::npos)
      what.erase(0, colon + 2);
    const std::string &st = lines[i + 1];
    if (st != "unsat" && st != "sat" && st != "unknown")
      throw std::runtime_error("smt: unexpected z3 status: " + st);
    r.push_back({std::move(what), st});
  }
  return r;
}

// Whether every check in CHECKS has status "unsat".
inline bool all_unsat(const std::vector<check_result> &checks) {
  if (checks.empty())
    return false;
  for (const check_result &c : checks)
    if (c.status != "unsat")
      return false;
  return true;
}

// The status of the first check in CHECKS whose description starts with
// PREFIX, or "" if none does.
inline std::string status_of(const std::vector<check_result> &checks,
                             std::string_view prefix) {
  for (const check_result &c : checks)
    if (std::string_view(c.what).starts_with(prefix))
      return c.status;
  return "";
}

// Whether every check in CHECKS whose description starts with PREFIX has
// status EXPECTED, at least one matches, and every other check is "unsat".
inline bool only_at(const std::vector<check_result> &checks,
                    std::string_view prefix,
                    std::string_view expected = "sat") {
  bool matched = false;
  for (const check_result &c : checks) {
    if (std::string_view(c.what).starts_with(prefix)) {
      if (c.status != expected)
        return false;
      matched = true;
    } else if (c.status != "unsat") {
      return false;
    }
  }
  return matched;
}

} // namespace smt

#endif
