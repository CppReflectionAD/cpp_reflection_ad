// nn/autodiff.hpp — reverse-mode AD of tensor-level code via body reflection.
//
//   using E = nn::ad::engine<nn::tensor_rule_set, nn::ad::cmath_rule_set>;
//   auto [loss, grads] = E::value_and_grad<^^my_loss, 0, 1>(W, b, x, t);
//
// reflects the body of `my_loss`, builds a DAG at compile time, and emits a
// forward sweep (the primal) and a reverse sweep (the adjoints) as straight
// C++: each node is one call or one built-in operator, unrolled with an
// expansion statement. Nothing is recorded at run time; there is no tape
// object and no virtual call, and the engine itself allocates nothing.
// nn/nn.hpp gives nn::ad::value_and_grad etc. with the default rule sets.
//
// What the engine reads in a body:
//   - `T v = expr;` declarations and a single `return expr;` (straight-line
//     code; control flow is not traversable in the proposal yet),
//   - parameters, locals, literals and constant variables,
//   - built-in + - * / and unary - on arithmetic types,
//   - calls. A call to a *primitive* -- a function one of the engine's rules
//     lists -- becomes one node. A call to anything else is inlined: the
//     engine reflects the callee's body into the same DAG. So user layers, and
//     the tensor operators (thin wrappers over primitives), need no rule.
//
// A rule is a class listing the functions it covers, by reflection (a
// function template covers all its specializations), and giving the
// vector-Jacobian product per argument:
//
//   struct matvec_rule {
//     static constexpr std::span<const nn::ad::info> callees =
//         nn::ad::listed({^^nn::matvec});
//     template <std::size_t I>
//     static auto pullback(const auto &g, const auto &y, const auto &W,
//                          const auto &x);   // dL/d(arg I), given g = dL/dy
//   };
//
// A function that can't be named, such as one overload of std::exp, is listed
// through a probe (see probe_callee). A unary scalar rule can give dy/dx in
// place of pullback: `d_x(x)` from the argument, or `d_y(y)` from the result.
// The engine takes rules and rule_sets of them; the first rule listing a
// callee wins.
//
// The engine knows no tensor type. Values live in a std::tuple of the node
// types (from the callees' return types and the parameter types: there is no
// type_of(expression)), and adjoints are summed with `+`. An adjoint's first
// contribution is assigned; a parameter none reaches gets
// `zeros_like(argument)`, found by ADL, or `T{}` if there is none.
//
// Activity: only the parameters listed in `Wrt` are differentiated. A node is
// "varied" if it depends on one of them; the reverse sweep visits only varied
// nodes and asks a rule only for the arguments that are varied. A node whose
// value never reaches the result is not evaluated at all.
//
// Compilers: Clang (p2996 fork) and GCC (with -freflection).

#ifndef NN_AUTODIFF_HPP
#define NN_AUTODIFF_HPP

#include <ad/meta.h>

#include <cstddef>
#include <initializer_list>
#include <meta>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace nn::ad {

// ::ad::meta, not ad::meta: inside nn::ad, that would name nn::ad::meta.
namespace m = ::ad::meta;
using info = m::info;

// A list of rules (or of rule_sets) to pass to an engine as one.
template <class... Rules> struct rule_set {};

// The values a rule's pullback reads besides the adjoint, declared in the
// rule as `static constexpr nn::ad::reads reads = ...;` (default: all). A value
// nothing reads is not computed by grad().
enum class reads : unsigned char { all, result, args, nothing };

// The function called by the one call in a probe's body. `^^std::exp` names an
// overload set, so a rule recovers each overload from a probe such as
// `template <class T> T p_exp(T x) { return std::exp(x); }`.
consteval info probe_callee(info probe) {
  for (info s : m::statements_of(m::body_of(probe)))
    if (m::is_return_statement(s)) {
      info e = m::return_operand_of(s);
      while (m::is_conversion(e))
        e = m::operands_of(e)[0];
      return m::declaration_of(m::callee_of(e));
    }
  throw "nn::ad: a probe is `return f(args...);`";
}

// A rule's `callees`: functions, or function templates for all their
// specializations.
consteval std::span<const info> listed(std::initializer_list<info> fns) {
  return std::define_static_array(std::vector<info>(fns));
}

// A rule's `callees`, from probes: one per overload it covers.
consteval std::span<const info> probed(std::initializer_list<info> probes) {
  std::vector<info> fns;
  for (info p : probes)
    fns.push_back(probe_callee(p));
  return std::define_static_array(fns);
}

enum class Kind : unsigned char {
  Input, // parameter `input` of the differentiated function
  Const, // literal or constant variable; value in `leaf`
  Call,  // primitive call: `fn(args...)`, adjoints from `rule`
  Add,   // built-in arithmetic
  Sub,
  Mul,
  Div,
  Neg,
  Convert, // arithmetic conversion to `type` (argument / return binding)
  Output,  // the return value
};

// The most arguments a primitive may take.
inline constexpr std::size_t max_args = 4;

struct Node {
  Kind kind = Kind::Input;
  std::size_t self = 0;  // slot, == index in the tape
  info type = ^^void;    // value type, cv-ref removed
  info fn = ^^void;      // Call: the callee
  info rule = ^^void;    // Call: the rule class
  info leaf = ^^void;    // Const: reflection of the value
  std::size_t input = 0; // Input: parameter index
  std::size_t nargs = 0;
  std::size_t arg[max_args] = {};
  // Analysis (see analyse()).
  bool varied = false;            // depends on a differentiated input
  bool live = false;              // its value reaches the result
  bool needed = false;            // the reverse sweep reads its value
  bool arg_varied[max_args] = {}; // per argument: varied
};

namespace detail {

inline constexpr std::size_t npos = static_cast<std::size_t>(-1);

template <class A, class B>
using add_t = decltype(std::declval<A>() + std::declval<B>());
template <class A, class B>
using sub_t = decltype(std::declval<A>() - std::declval<B>());
template <class A, class B>
using mul_t = decltype(std::declval<A>() * std::declval<B>());
template <class A, class B>
using div_t = decltype(std::declval<A>() / std::declval<B>());
template <class A> using neg_t = decltype(-std::declval<A>());

struct Ctx {
  std::vector<Node> nodes;
  // Names in scope -> slots. Entries from `scope` on belong to the body being
  // lowered; an inlined callee does not see its caller's locals.
  std::vector<info> env_decl;
  std::vector<std::size_t> env_slot;
  std::size_t scope = 0;
  std::vector<info> call_stack; // functions being lowered (recursion guard)
  std::vector<info> listed_fn, listed_rule; // from the engine's rules
};

consteval std::size_t emit(Ctx &c, Node n) {
  n.self = c.nodes.size();
  c.nodes.push_back(n);
  return n.self;
}

consteval info value_type(info t) { return m::remove_cvref(t); }

// Peel the conversions (lvalue-to-rvalue, derived-to-base, a copy of a class
// value, ...) to reach the expression that computes. Arithmetic conversions
// are restored where the type is known (coerce()), since a conversion's target
// type cannot be queried.
consteval info strip(info e) {
  while (m::is_expression(e) && m::is_conversion(e))
    e = m::operands_of(e)[0];
  return e;
}

consteval std::size_t lookup(const Ctx &c, std::string_view name) {
  for (std::size_t i = c.env_decl.size(); i-- > c.scope;)
    if (m::has_identifier(c.env_decl[i]) &&
        m::identifier_of(c.env_decl[i]) == name)
      return c.env_slot[i];
  return npos;
}

consteval void bind(Ctx &c, info decl, std::size_t slot) {
  c.env_decl.push_back(decl);
  c.env_slot.push_back(slot);
}

// Make `slot` a value of type `to`: as is, or through an arithmetic Convert.
// Anything else would be a conversion the engine cannot see into.
consteval std::size_t coerce(Ctx &c, std::size_t slot, info to) {
  const info from = c.nodes[slot].type;
  if (m::is_same_type(from, to))
    return slot;
  if (m::is_arithmetic_type(from) && m::is_arithmetic_type(to))
    return emit(
        c, Node{.kind = Kind::Convert, .type = to, .nargs = 1, .arg = {slot}});
  throw "nn::ad: a value is used as a different class type; only arithmetic "
        "conversions are modelled";
}

// The member of class `cls` called `name`, or a null reflection.
consteval info member_named(info cls, std::string_view name) {
  for (info mem : m::members_of(cls, m::access_context::unchecked()))
    if (m::has_identifier(mem) && m::identifier_of(mem) == name)
      return mem;
  return info{};
}

// Every (callee, rule) pair the rules in `rules` (a rule or a rule_set) list,
// in order. By reference: extract by value needs a structural type, and
// std::span is not one.
consteval void list_rules(Ctx &c, info rules) {
  rules = m::dealias(rules);
  if (m::has_template_arguments(rules) && m::template_of(rules) == ^^rule_set) {
    for (info r : m::template_arguments_of(rules))
      list_rules(c, r);
    return;
  }
  const info cs = m::is_class_type(rules) && m::is_complete_type(rules)
                      ? member_named(rules, "callees")
                      : info{};
  if (cs == info{})
    throw "nn::ad: a rule is a complete class with "
          "`static constexpr std::span<const info> callees`";
  for (info f : m::extract<const std::span<const info> &>(cs)) {
    c.listed_fn.push_back(f);
    c.listed_rule.push_back(rules);
  }
}

// The first rule listing `callee`, or its template for a specialization (so
// one rule covers every shape); else a null reflection.
consteval info rule_for(const Ctx &c, info callee) {
  const info tmpl =
      m::has_template_arguments(callee) ? m::template_of(callee) : info{};
  for (std::size_t i = 0; i < c.listed_fn.size(); ++i)
    if (c.listed_fn[i] == callee || c.listed_fn[i] == tmpl)
      return c.listed_rule[i];
  return info{};
}

consteval std::size_t lower(Ctx &c, info e);
consteval std::size_t lower_body(Ctx &c, info body);

consteval std::size_t inline_call(Ctx &c, info callee,
                                  const std::vector<std::size_t> &args) {
  for (info f : c.call_stack)
    if (f == callee)
      throw "nn::ad: recursive calls cannot be inlined";

  const std::size_t outer_scope = c.scope;
  const std::size_t mark = c.env_decl.size();
  const auto params = m::parameters_of(callee);
  for (std::size_t i = 0; i < params.size(); ++i)
    bind(c, params[i], args[i]);
  c.scope = mark;

  c.call_stack.push_back(callee);
  const std::size_t ret = lower_body(c, m::body_of(callee));
  c.call_stack.pop_back();

  c.env_decl.resize(mark);
  c.env_slot.resize(mark);
  c.scope = outer_scope;
  return coerce(c, ret, value_type(m::return_type_of(callee)));
}

consteval std::size_t lower_call(Ctx &c, info call) {
  const info callee = m::declaration_of(m::callee_of(call));
  if (!m::is_function(callee))
    throw "nn::ad: only calls to named functions are supported";
  if (m::is_class_member(callee) && !m::is_static_member(callee))
    throw "nn::ad: member function calls are not supported";

  const auto ops = m::operands_of(call); // the arguments, without the callee
  const auto params = m::parameters_of(callee);
  if (ops.size() != params.size())
    throw "nn::ad: calls must pass every argument explicitly (no defaults, no "
          "varargs)";

  std::vector<std::size_t> args;
  for (std::size_t i = 0; i < params.size(); ++i)
    args.push_back(
        coerce(c, lower(c, ops[i]), value_type(m::type_of(params[i]))));

  const info rule = rule_for(c, callee);
  if (rule == info{})
    return inline_call(c, callee, args);

  if (args.size() > max_args)
    throw "nn::ad: a primitive takes at most max_args arguments";
  if (args.size() != 1 && (member_named(rule, "d_x") != info{} ||
                           member_named(rule, "d_y") != info{}))
    throw "nn::ad: a rule giving d_x or d_y must be for a unary function; "
          "give pullback<I> instead";
  Node n{.kind = Kind::Call,
         .type = value_type(m::return_type_of(callee)),
         .fn = callee,
         .rule = rule,
         .nargs = args.size()};
  for (std::size_t i = 0; i < args.size(); ++i)
    n.arg[i] = args[i];
  return emit(c, n);
}

consteval std::size_t lower(Ctx &c, info e) {
  e = strip(e);

  if (m::is_id_expression(e)) {
    const info decl = m::declaration_of(e);
    // Only a function's own names can be parameters or locals (not `k::t`).
    if (m::has_identifier(decl) && m::is_function(m::parent_of(decl)))
      if (std::size_t s = lookup(c, m::identifier_of(decl)); s != npos)
        return s;
    // Not a parameter or local: a constant (e.g. a constexpr global), folded.
    const info v = m::constant_of(decl);
    return emit(c, Node{.kind = Kind::Const,
                        .type = value_type(m::type_of(v)),
                        .leaf = v});
  }

  if (m::is_literal(e)) {
    const info v = m::constant_of(e);
    return emit(c, Node{.kind = Kind::Const,
                        .type = value_type(m::type_of(v)),
                        .leaf = v});
  }

  // Before the operators: an overloaded operator is both a call and an
  // operator expression.
  if (m::is_function_call(e))
    return lower_call(c, e);

  // Built-in operators.
  if (m::is_operator_expression(e) && m::operands_of(e).size() == 1) {
    const m::operators op = m::expression_operator_of(e);
    const std::size_t a = lower(c, m::operands_of(e)[0]);
    if (!m::is_arithmetic_type(c.nodes[a].type))
      throw "nn::ad: built-in operator on a non-arithmetic value";
    if (op == m::operators::op_plus)
      return a;
    if (op == m::operators::op_minus) {
      const info t = m::substitute(^^neg_t, {
                                                c.nodes[a].type});
      return emit(c, Node{.kind = Kind::Neg,
                          .type = m::dealias(t),
                          .nargs = 1,
                          .arg = {a}});
    }
    throw "nn::ad: unsupported unary operator";
  }

  if (m::is_operator_expression(e) && m::operands_of(e).size() == 2) {
    const auto ops = m::operands_of(e);
    const m::operators op = m::expression_operator_of(e);
    const std::size_t a = lower(c, ops[0]);
    const std::size_t b = lower(c, ops[1]);
    const info ta = c.nodes[a].type, tb = c.nodes[b].type;
    if (!m::is_arithmetic_type(ta) || !m::is_arithmetic_type(tb))
      throw "nn::ad: built-in operator on a non-arithmetic value";
    Kind k;
    info alias;
    if (op == m::operators::op_plus) {
      k = Kind::Add;
      alias = ^^add_t;
    } else if (op == m::operators::op_minus) {
      k = Kind::Sub;
      alias = ^^sub_t;
    } else if (op == m::operators::op_star) {
      k = Kind::Mul;
      alias = ^^mul_t;
    } else if (op == m::operators::op_slash) {
      k = Kind::Div;
      alias = ^^div_t;
    } else {
      throw "nn::ad: unsupported binary operator";
    }
    return emit(c, Node{.kind = k,
                        .type = m::dealias(m::substitute(alias, {ta, tb})),
                        .nargs = 2,
                        .arg = {a, b}});
  }

  throw "nn::ad: unsupported expression (conditional operators, casts the "
        "engine cannot see through, constructions, ...)";
}

// Lower a straight-line body; returns the slot of its return value.
consteval std::size_t lower_body(Ctx &c, info body) {
  // Check the shape first, so a loop is reported as such rather than as
  // whatever expression precedes it.
  for (info s : m::statements_of(body))
    if (!m::is_declaration_statement(s) && !m::is_return_statement(s))
      throw "nn::ad: unsupported statement; bodies are `T v = expr;` "
            "declarations and one `return expr;`. A function with loops or "
            "branches needs a rule (make it a primitive).";

  std::size_t ret = npos;
  for (info s : m::statements_of(body)) {
    if (ret != npos)
      throw "nn::ad: statement after the return statement";
    if (m::is_declaration_statement(s)) {
      const auto decls = m::declarators_of(s);
      const auto inits = m::initializers_of(s);
      for (std::size_t d = 0; d < decls.size(); ++d) {
        const info v = decls[d];
        const std::size_t slot =
            coerce(c, lower(c, inits[d]), value_type(m::type_of(v)));
        bind(c, v, slot);
      }
    } else {
      ret = lower(c, m::return_operand_of(s));
    }
  }
  if (ret == npos)
    throw "nn::ad: function body has no return statement";
  return ret;
}

// A rule's `reads`; otherwise implied by d_y / d_x, else all.
consteval reads reads_of(info rule) {
  if (const info r = member_named(rule, "reads"); r != info{})
    return m::extract<reads>(r);
  if (member_named(rule, "d_y") != info{})
    return reads::result;
  if (member_named(rule, "d_x") != info{})
    return reads::args;
  return reads::all;
}

// Varied: forward, from the differentiated inputs. Live: backward, from the
// output. Needed: backward, from the values the reverse sweep reads, through
// the values they are computed from. Nodes are in topological order
// (arguments before their users).
consteval void analyse(std::vector<Node> &ns, unsigned long long mask) {
  for (Node &n : ns) {
    if (n.kind == Kind::Input)
      n.varied = (mask >> n.input) & 1ull;
    for (std::size_t i = 0; i < n.nargs; ++i) {
      n.arg_varied[i] = ns[n.arg[i]].varied;
      n.varied = n.varied || n.arg_varied[i];
    }
  }
  ns.back().live = true;
  for (std::size_t i = ns.size(); i-- > 0;)
    if (ns[i].live)
      for (std::size_t k = 0; k < ns[i].nargs; ++k)
        ns[ns[i].arg[k]].live = true;

  for (Node &n : ns) {
    if (!n.live || !n.varied)
      continue;
    const reads r = n.kind == Kind::Call ? reads_of(n.rule)
                    : n.kind == Kind::Mul || n.kind == Kind::Div
                        ? reads::args
                        : reads::nothing;
    if (r == reads::all || r == reads::result)
      n.needed = true;
    if (r == reads::all || r == reads::args)
      for (std::size_t k = 0; k < n.nargs; ++k)
        ns[n.arg[k]].needed = true;
  }
  for (std::size_t i = ns.size(); i-- > 0;)
    if (ns[i].needed)
      for (std::size_t k = 0; k < ns[i].nargs; ++k)
        ns[ns[i].arg[k]].needed = true;
}

consteval std::vector<Node> build(info fn, unsigned long long mask,
                                  info rules) {
  Ctx c;
  list_rules(c, rules);
  const auto params = m::parameters_of(fn);
  if (params.size() > 64)
    throw "nn::ad: at most 64 parameters";
  for (std::size_t i = 0; i < params.size(); ++i)
    bind(c, params[i],
         emit(c, Node{.kind = Kind::Input,
                      .type = value_type(m::type_of(params[i])),
                      .input = i}));

  c.call_stack.push_back(fn);
  std::size_t ret = lower_body(c, m::body_of(fn));
  const info rt = value_type(m::return_type_of(fn));
  if (!m::is_arithmetic_type(rt))
    throw "nn::ad: the differentiated function must return a scalar";
  ret = coerce(c, ret, rt);
  emit(c, Node{.kind = Kind::Output, .type = rt, .nargs = 1, .arg = {ret}});

  analyse(c.nodes, mask);
  return c.nodes;
}

consteval std::vector<Node> reversed(std::span<const Node> ns) {
  return std::vector<Node>(ns.rbegin(), ns.rend());
}

consteval info storage_of(std::span<const Node> ns) {
  std::vector<info> types;
  for (const Node &n : ns)
    types.push_back(n.type);
  return m::substitute(^^std::tuple, types);
}

// The first contribution to an adjoint is assigned, so an adjoint never needs
// a zero of its value's shape (a value grad() skips has none).
template <std::size_t I, class Vals, class G>
constexpr void accumulate(Vals &adj, bool *seen, const G &g) {
  auto &acc = std::get<I>(adj);
  if (seen[I]) {
    acc = acc + g;
  } else {
    acc = g;
    seen[I] = true;
  }
}

template <class T> constexpr T zero_like(const T &v) {
  if constexpr (requires { zeros_like(v); })
    return zeros_like(v);
  else
    return T{};
}

template <Node N, class Vals, class In>
constexpr void forward(Vals &val, const In &in) {
  auto &out = std::get<N.self>(val);
  if constexpr (N.kind == Kind::Input) {
    out = std::get<N.input>(in);
  } else if constexpr (N.kind == Kind::Const) {
    out = [:N.leaf:];
  } else if constexpr (N.kind == Kind::Call) {
    [&]<std::size_t... K>(std::index_sequence<K...>) {
      out = [:N.fn:](std::get<N.arg[K]>(val)...);
    }(std::make_index_sequence<N.nargs>{});
  } else {
    const auto &a = std::get<N.arg[0]>(val);
    if constexpr (N.kind == Kind::Output || N.kind == Kind::Convert)
      out = static_cast<typename[:N.type:]>(a);
    else if constexpr (N.kind == Kind::Neg)
      out = -a;
    else {
      const auto &b = std::get<N.arg[1]>(val);
      if constexpr (N.kind == Kind::Add)
        out = a + b;
      else if constexpr (N.kind == Kind::Sub)
        out = a - b;
      else if constexpr (N.kind == Kind::Mul)
        out = a * b;
      else
        out = a / b;
    }
  }
}

// dL/d(argument K) of a primitive call, from its rule.
template <Node N, std::size_t K, class Vals>
constexpr void pull(const Vals &val, Vals &adj, bool *seen) {
  if constexpr (N.arg_varied[K]) {
    using Rule = typename[:N.rule:];
    const auto &g = std::get<N.self>(adj);
    const auto &y = std::get<N.self>(val);
    const auto &x = std::get<N.arg[0]>(val);
    if constexpr (requires { Rule::d_y(y); }) {
      accumulate<N.arg[K]>(adj, seen, g * Rule::d_y(y));
    } else if constexpr (requires { Rule::d_x(x); }) {
      accumulate<N.arg[K]>(adj, seen, g * Rule::d_x(x));
    } else {
      [&]<std::size_t... J>(std::index_sequence<J...>) {
        accumulate<N.arg[K]>(
            adj, seen,
            Rule::template pullback<K>(g, y, std::get<N.arg[J]>(val)...));
      }(std::make_index_sequence<N.nargs>{});
    }
  }
}

template <Node N, class Vals>
constexpr void backward(const Vals &val, Vals &adj, bool *seen) {
  if (!seen[N.self])
    return; // no contribution reached it: its adjoint is zero
  const auto &g = std::get<N.self>(adj);
  [[maybe_unused]] constexpr std::size_t a0 = N.arg[0], a1 = N.arg[1];
  if constexpr (N.kind == Kind::Call) {
    [&]<std::size_t... K>(std::index_sequence<K...>) {
      (pull<N, K>(val, adj, seen), ...);
    }(std::make_index_sequence<N.nargs>{});
  } else if constexpr (N.kind == Kind::Output || N.kind == Kind::Convert) {
    using A = std::tuple_element_t<a0, Vals>;
    accumulate<a0>(adj, seen, static_cast<A>(g));
  } else if constexpr (N.kind == Kind::Neg) {
    accumulate<a0>(adj, seen, -g);
  } else if constexpr (N.kind == Kind::Add || N.kind == Kind::Sub ||
                       N.kind == Kind::Mul || N.kind == Kind::Div) {
    const auto &a = std::get<a0>(val);
    const auto &b = std::get<a1>(val);
    if constexpr (N.kind == Kind::Add) {
      if constexpr (N.arg_varied[0])
        accumulate<a0>(adj, seen, g);
      if constexpr (N.arg_varied[1])
        accumulate<a1>(adj, seen, g);
    } else if constexpr (N.kind == Kind::Sub) {
      if constexpr (N.arg_varied[0])
        accumulate<a0>(adj, seen, g);
      if constexpr (N.arg_varied[1])
        accumulate<a1>(adj, seen, -g);
    } else if constexpr (N.kind == Kind::Mul) {
      if constexpr (N.arg_varied[0])
        accumulate<a0>(adj, seen, g * b);
      if constexpr (N.arg_varied[1])
        accumulate<a1>(adj, seen, g * a);
    } else {
      if constexpr (N.arg_varied[0])
        accumulate<a0>(adj, seen, g / b);
      if constexpr (N.arg_varied[1])
        accumulate<a1>(adj, seen, -(g * a / (b * b)));
    }
  }
}

// The gradient for parameter W, in its node's type. A parameter no adjoint
// reached (unused, or not on the result's path) gets a zero of its argument's
// shape.
template <std::size_t W, class Vals, class In>
auto gradient_of(Vals &adj, const bool *seen, const In &in) {
  using T = std::tuple_element_t<W, Vals>;
  if (seen[W])
    return std::move(std::get<W>(adj));
  return static_cast<T>(zero_like(std::get<W>(in)));
}

// The sweep behind value_and_grad (Value) and grad: grad computes only the
// values the reverse sweep reads.
template <info Fn, bool Value, class Rules, std::size_t... Wrt, class... Args>
auto sweep(const Args &...args) {
  static_assert(sizeof...(Wrt) > 0,
                "nn::ad: name the parameters to "
                "differentiate, e.g. value_and_grad<^^f, 0>");
  static_assert(sizeof...(Args) == m::parameters_of(Fn).size(),
                "nn::ad: wrong number of arguments");
  static_assert(((Wrt < sizeof...(Args)) && ...),
                "nn::ad: a Wrt index is not a parameter of Fn");
  // An adjoint has its node's type, so an integer one would truncate.
  static_assert(
      (!m::is_integral_type(m::remove_cvref(m::type_of(
           m::parameters_of(Fn)[Wrt < sizeof...(Args) ? Wrt : 0]))) &&
       ...),
      "nn::ad: a differentiated parameter must not have an integer type");
  constexpr unsigned long long mask = ((1ull << Wrt) | ... | 0ull);
  static constexpr auto nodes =
      std::define_static_array(build(Fn, mask, ^^Rules));
  static constexpr auto rnodes = std::define_static_array(reversed(nodes));
  constexpr std::size_t out = nodes.size() - 1;
  using Vals = typename[:storage_of(nodes):];

  const auto in = std::forward_as_tuple(args...);
  Vals val{};
  Vals adj{};
  bool seen[nodes.size()] = {};

  template for (constexpr Node n : nodes) {
    if constexpr (Value ? n.live : n.needed)
      forward<n>(val, in);
  }

  std::get<out>(adj) = 1;
  seen[out] = true;
  template for (constexpr Node n : rnodes) {
    if constexpr (n.live && n.varied)
      backward<n>(val, adj, seen);
  }

  auto grads = std::tuple{gradient_of<Wrt>(adj, seen, in)...};
  if constexpr (Value)
    return std::pair{std::get<out>(val), std::move(grads)};
  else
    return grads;
}

} // namespace detail

template <std::size_t... Wrt>
inline constexpr unsigned long long mask_of = ((1ull << Wrt) | ... | 0ull);

// Differentiation with the primitives that `Rules` (rules and rule_sets of
// them) list; the first rule listing a callee wins.
template <class... Rules> struct engine {
  // The DAG of Fn, analysed for differentiation w.r.t. the parameters in
  // `Mask` (bit i = parameter i). Exposed for inspection and tests.
  template <info Fn, unsigned long long Mask>
  static constexpr std::span<const Node> tape =
      std::define_static_array(detail::build(Fn, Mask, ^^rule_set<Rules...>));

  // Fn's value and its gradient w.r.t. parameters Wrt..., as
  // pair{value, tuple{dFn/d(param Wrt)...}}. A gradient has its parameter's
  // type.
  template <info Fn, std::size_t... Wrt, class... Args>
  static auto value_and_grad(const Args &...args) {
    return detail::sweep<Fn, true, rule_set<Rules...>, Wrt...>(args...);
  }

  // Just the gradient: tuple{dFn/d(param Wrt)...}. Skips the values the
  // reverse sweep doesn't read, such as the result itself.
  template <info Fn, std::size_t... Wrt, class... Args>
  static auto grad(const Args &...args) {
    return detail::sweep<Fn, false, rule_set<Rules...>, Wrt...>(args...);
  }
};

} // namespace nn::ad

#endif // NN_AUTODIFF_HPP
