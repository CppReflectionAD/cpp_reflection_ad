// nn/autodiff.hpp — reverse-mode AD of tensor-level code via body reflection.
//
//   auto [loss, grads] = nn::ad::value_and_grad<^^my_loss, 0, 1>(W, b, x, t);
//
// reflects the body of `my_loss`, builds a DAG at compile time, and emits a
// forward sweep (the primal) and a reverse sweep (the adjoints) as straight
// C++: each node is one call or one built-in operator, unrolled with an
// expansion statement. Nothing is recorded at run time; there is no tape
// object, no virtual call, no allocation.
//
// What the engine reads in a body:
//   - `T v = expr;` declarations and a single `return expr;` (straight-line
//     code; control flow is not traversable in the proposal yet),
//   - parameters, locals, literals and constant variables,
//   - built-in + - * / and unary - on arithmetic types,
//   - calls. A call to a *primitive* -- a function with a registered rule,
//     `nn::ad::vjp<^^f>` -- becomes one node. A call to anything else is
//     inlined: the engine reflects the callee's body into the same DAG. So
//     user layers, and the tensor operators (thin wrappers over primitives),
//     need no rule.
//
// A rule is keyed on the callee's reflection (on its template, for a function
// template specialization), not on its name, and gives the vector-Jacobian
// product per argument:
//
//   template <> struct nn::ad::vjp<nn::ad::key(^^nn::matvec)> {
//     template <std::size_t I>
//     static auto pullback(const auto &g, const auto &y, const auto &W,
//                          const auto &x);   // dL/d(arg I), given g = dL/dy
//   };
//
// The engine knows no tensor type. Values live in a std::tuple of the node
// types (from the callees' return types and the parameter types: there is no
// type_of(expression)), and adjoints are summed with `+`.
//
// Activity: only the parameters listed in `Wrt` are differentiated. A node is
// "varied" if it depends on one of them; the reverse sweep visits only varied
// nodes and asks a rule only for the arguments that are varied. A node whose
// value never reaches the result is not evaluated at all.
//
// Compilers: Clang (p2996 fork) and GCC (with -freflection), each of which
// reflects some constructs differently -- see strip() below.

#ifndef NN_AUTODIFF_HPP
#define NN_AUTODIFF_HPP

#include <ad/meta.h>

#include <cstddef>
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

// The registry key of a function: the function, or for a template or one of
// its specializations, the template. Taken from members_of of the enclosing
// scope, so that every route to the same entity gives one representation:
// GCC compares `template_of(spec) == ^^nn::matvec` true, yet treats the two as
// different template arguments, so `vjp<^^nn::matvec>` would never be found
// from a call. (Clang needs no normalisation; it does no harm there.)
consteval info key(info fn) {
  if (m::has_template_arguments(fn))
    fn = m::template_of(fn);
  for (info mem :
       m::members_of(m::parent_of(fn), m::access_context::unchecked()))
    if (mem == fn)
      return mem;
  return fn; // not a listed member (e.g. a hidden friend): use it as is
}

// The rule registry. Declared, never defined: a specialization makes its key a
// primitive. Register with the normalised key:
//   template <> struct nn::ad::vjp<nn::ad::key(^^nn::matvec)> { ... };
template <info Key> struct vjp;

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
  info rule = ^^void;    // Call: the vjp<> specialization
  info leaf = ^^void;    // Const: reflection of the value
  std::size_t input = 0; // Input: parameter index
  std::size_t nargs = 0;
  std::size_t arg[max_args] = {};
  // Analysis (see analyse()).
  bool varied = false;            // depends on a differentiated input
  bool live = false;              // its value reaches the result
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

struct Ctx {
  std::vector<Node> nodes;
  // Names in scope -> slots. Entries from `scope` on belong to the body being
  // lowered; an inlined callee does not see its caller's locals.
  std::vector<info> env_decl;
  std::vector<std::size_t> env_slot;
  std::size_t scope = 0;
  std::vector<info> call_stack; // functions being lowered (recursion guard)
};

consteval std::size_t emit(Ctx &c, Node n) {
  n.self = c.nodes.size();
  c.nodes.push_back(n);
  return n.self;
}

consteval info value_type(info t) { return m::remove_cvref(t); }

// Peel the nodes that carry no computation, to reach the expression that does.
// What wraps an operand differs by compiler:
//   - both: implicit conversions (lvalue-to-rvalue, derived-to-base, ...) are
//     conversions. Arithmetic conversions are restored where the type is known
//     (coerce()), since a conversion's target type cannot be queried.
//   - Clang: a copy or move of a class value is a one-operand construct.
//   - GCC: binding to a `const T &` parameter is a conversion of `&operand`; a
//     temporary, and the return slot, is a construct of an unnamed variable
//     from its initializer. If the class has a non-trivial destructor, a
//     temporary's construct has a third operand: the destructor call that
//     cleans it up.
consteval info strip(info e) {
  while (m::is_expression(e)) {
    if (m::is_conversion(e)) {
      e = m::operands_of(e)[0];
      if (m::is_expression(e) && m::is_operator_expression(e) &&
          m::operands_of(e).size() == 1 &&
          m::expression_operator_of(e) == m::operators::op_ampersand)
        e = m::operands_of(e)[0];
      continue;
    }
    if (m::is_construct(e)) {
      auto ops = m::operands_of(e);
      if (ops.size() == 1) {
        e = ops[0];
        continue;
      }
      // The cleanup is a call on the temporary (`&tmp`). Its callee is GCC's
      // destructor clone, for which is_destructor() is false, so match the
      // operand rather than the callee.
      const auto cleans_up = [&](info call) {
        if (!m::is_expression(call) || !m::is_function_call(call))
          return false;
        const auto args = m::operands_of(call);
        if (args.size() != 1 || !m::is_expression(args[0]) ||
            !m::is_operator_expression(args[0]) ||
            m::expression_operator_of(args[0]) != m::operators::op_ampersand)
          return false;
        const info target = m::operands_of(args[0])[0];
        return m::is_expression(target) && m::is_id_expression(target) &&
               m::declaration_of(target) == m::declaration_of(ops[0]);
      };
      if ((ops.size() == 2 || ops.size() == 3) && m::is_expression(ops[0]) &&
          m::is_id_expression(ops[0]) &&
          !m::has_identifier(m::declaration_of(ops[0])) &&
          (ops.size() == 2 || cleans_up(ops[2]))) {
        e = ops[1];
        continue;
      }
    }
    break;
  }
  return e;
}

consteval std::size_t lookup(const Ctx &c, std::string_view name) {
  for (std::size_t i = c.env_decl.size(); i-- > c.scope;)
    if (m::identifier_of(c.env_decl[i]) == name)
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

// vjp<key(callee)> if it is defined, else a null reflection. The key of a
// function template specialization is its template, so one rule covers every
// shape.
consteval info rule_for(info callee) {
  const info spec = m::substitute(^^vjp, {
                                             m::reflect_constant(key(callee))});
  return m::is_complete_type(spec) ? spec : info{};
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

  const info rule = rule_for(callee);
  if (rule == info{})
    return inline_call(c, callee, args);

  if (args.size() > max_args)
    throw "nn::ad: a primitive takes at most max_args arguments";
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
    if (m::has_identifier(decl))
      if (std::size_t s = lookup(c, m::identifier_of(decl)); s != npos)
        return s;
    // Not a parameter or local: a constant (e.g. a constexpr global), folded.
    const info v = m::constant_of(e);
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

  // Before the operators: clang reports an overloaded operator as both a call
  // and an operator expression, GCC as a call only.
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
    if (op == m::operators::op_minus)
      return emit(c, Node{.kind = Kind::Neg,
                          .type = c.nodes[a].type,
                          .nargs = 1,
                          .arg = {a}});
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
            "branches needs a vjp rule (make it a primitive).";

  const std::size_t begin = c.env_decl.size();
  std::size_t ret = npos;
  for (info s : m::statements_of(body)) {
    if (ret != npos)
      throw "nn::ad: statement after the return statement";
    if (m::is_declaration_statement(s)) {
      const auto decls = m::declarators_of(s);
      const auto inits = m::initializers_of(s);
      for (std::size_t d = 0; d < decls.size(); ++d) {
        const info v = decls[d];
        // GCC lists each declaration statement twice. A block declares a
        // name once, so a name already declared in this body is the same
        // statement.
        bool seen = false;
        for (std::size_t i = begin; i < c.env_decl.size(); ++i)
          seen = seen || m::identifier_of(c.env_decl[i]) == m::identifier_of(v);
        if (seen)
          continue;
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

// Varied: forward, from the differentiated inputs. Live: backward, from the
// output. Nodes are in topological order (arguments before their users).
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
}

consteval std::vector<Node> build(info fn, unsigned long long mask) {
  Ctx c;
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

template <class T, class G> constexpr void accumulate(T &acc, const G &g) {
  acc = acc + g;
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
constexpr void pull(const Vals &val, Vals &adj) {
  if constexpr (N.arg_varied[K]) {
    using Rule = typename[:N.rule:];
    [&]<std::size_t... J>(std::index_sequence<J...>) {
      accumulate(std::get<N.arg[K]>(adj),
                 Rule::template pullback<K>(std::get<N.self>(adj),
                                            std::get<N.self>(val),
                                            std::get<N.arg[J]>(val)...));
    }(std::make_index_sequence<N.nargs>{});
  }
}

template <Node N, class Vals>
constexpr void backward(const Vals &val, Vals &adj) {
  const auto &g = std::get<N.self>(adj);
  if constexpr (N.kind == Kind::Call) {
    [&]<std::size_t... K>(std::index_sequence<K...>) {
      (pull<N, K>(val, adj), ...);
    }(std::make_index_sequence<N.nargs>{});
  } else if constexpr (N.kind == Kind::Output || N.kind == Kind::Convert) {
    using A = std::tuple_element_t<N.arg[0], Vals>;
    accumulate(std::get<N.arg[0]>(adj), static_cast<A>(g));
  } else if constexpr (N.kind == Kind::Neg) {
    accumulate(std::get<N.arg[0]>(adj), -g);
  } else if constexpr (N.kind == Kind::Add || N.kind == Kind::Sub ||
                       N.kind == Kind::Mul || N.kind == Kind::Div) {
    const auto &a = std::get<N.arg[0]>(val);
    const auto &b = std::get<N.arg[1]>(val);
    auto &ga = std::get<N.arg[0]>(adj);
    auto &gb = std::get<N.arg[1]>(adj);
    if constexpr (N.kind == Kind::Add) {
      if constexpr (N.arg_varied[0])
        accumulate(ga, g);
      if constexpr (N.arg_varied[1])
        accumulate(gb, g);
    } else if constexpr (N.kind == Kind::Sub) {
      if constexpr (N.arg_varied[0])
        accumulate(ga, g);
      if constexpr (N.arg_varied[1])
        accumulate(gb, -g);
    } else if constexpr (N.kind == Kind::Mul) {
      if constexpr (N.arg_varied[0])
        accumulate(ga, g * b);
      if constexpr (N.arg_varied[1])
        accumulate(gb, g * a);
    } else {
      if constexpr (N.arg_varied[0])
        accumulate(ga, g / b);
      if constexpr (N.arg_varied[1])
        accumulate(gb, -(g * a / (b * b)));
    }
  }
}

} // namespace detail

// The DAG of Fn, analysed for differentiation w.r.t. the parameters in `Mask`
// (bit i = parameter i). Exposed for inspection and tests.
template <info Fn, unsigned long long Mask>
inline constexpr std::span<const Node> tape =
    std::define_static_array(detail::build(Fn, Mask));

template <std::size_t... Wrt>
inline constexpr unsigned long long mask_of = ((1ull << Wrt) | ... | 0ull);

// Fn's value and its gradient w.r.t. parameters Wrt..., as
// pair{value, tuple{dFn/d(param Wrt)...}}. A gradient has its parameter's type.
template <info Fn, std::size_t... Wrt, class... Args>
auto value_and_grad(const Args &...args) {
  static_assert(sizeof...(Wrt) > 0,
                "nn::ad: name the parameters to "
                "differentiate, e.g. value_and_grad<^^f, 0>");
  static_assert(sizeof...(Args) == m::parameters_of(Fn).size(),
                "nn::ad: wrong number of arguments");
  constexpr unsigned long long mask = mask_of<Wrt...>;
  static constexpr auto nodes = tape<Fn, mask>;
  static constexpr auto rnodes =
      std::define_static_array(detail::reversed(nodes));
  constexpr std::size_t out = nodes.size() - 1;
  using Vals = typename[:detail::storage_of(nodes):];

  const auto in = std::forward_as_tuple(args...);
  Vals val{};
  Vals adj{}; // zero: value-initialized doubles and tensors

  template for (constexpr Node n : nodes) {
    if constexpr (n.live)
      detail::forward<n>(val, in);
  }

  std::get<out>(adj) = 1;
  template for (constexpr Node n : rnodes) {
    if constexpr (n.live && n.varied)
      detail::backward<n>(val, adj);
  }

  return std::pair{std::get<out>(val), std::tuple{std::get<Wrt>(adj)...}};
}

// Just the gradient: tuple{dFn/d(param Wrt)...}.
template <info Fn, std::size_t... Wrt, class... Args>
auto grad(const Args &...args) {
  return value_and_grad<Fn, Wrt...>(args...).second;
}

} // namespace nn::ad

#endif // NN_AUTODIFF_HPP
