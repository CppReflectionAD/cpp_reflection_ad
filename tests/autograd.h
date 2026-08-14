// autograd.h — compile-time automatic differentiation via reflection.
//
// Pick a straight-line scalar function, reflect its body, and get a zero-cost
// spliced derivative. Built on the statement/expression reflection extension to
// P2996 (body_of / statements_of / return_value_of / ... plus the existing
// expression reflection).
//
// Design (see docs/ + plan): reflection -> SSA/let-DAG IR -> array-tape codegen.
// Each statement of the body becomes SSA nodes; sweeps over runtime arrays are
// unrolled with an expansion statement, indexing the in-scope arrays directly
// (only literal leaves are spliced).
//
// Modes:
//   - forward_derivative<^^f, Wrt>(args...)  — forward mode (JVP), one partial.
//   - gradient_of<^^f>(args...)              — full gradient via P forward passes.
//   - gradient_reverse<^^f>(args...)         — full gradient in ONE reverse pass
//       (primal sweep + adjoint sweep over the reversed DAG with accumulation);
//       the efficient path for scalar-output, many-input functions.
//
// Activity analysis (mark_activity) runs on the DAG before codegen: it marks
// which values are "varied" (depend on a differentiated input) and which are
// "needed" (read by an emitted derivative), and the codegen elides everything
// else. This removes derivative-zero terms (no `x * 0`) and drops primals that
// feed only the function value -- e.g. d/dy of `x*y + exp(x)` emits just `x`,
// with no exp call.
//
// Scope: straight-line bodies (parameters, `T v = expr;` locals, one
// `return expr;`), scalar T, ops + - * / unary-minus and unary calls
// sin/cos/exp/log/sqrt. The core is rule-driven so more ops / tensors are
// additive.

#ifndef REFLECT_DEMO_AUTOGRAD_H
#define REFLECT_DEMO_AUTOGRAD_H

#include <meta>
#include <vector>
#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <utility>

namespace ad {
using std::meta::info;
namespace m = std::meta;

// ---------------------------------------------------------------------------
// IR
// ---------------------------------------------------------------------------
enum class OpKind {
  Input, Const, Output,
  Add, Sub, Mul, Div, Neg,
  Sin, Cos, Exp, Log, Sqrt,
  // Tensor ops (recognised as named calls; VJPs live in the tensor engine).
  Matmul, Transpose, Sum, Relu,
};

struct Node {
  OpKind op = OpKind::Input;
  std::size_t self = 0;      // this node's SSA slot (== its index)
  std::size_t a = 0, b = 0;  // operand slots
  info leaf = ^^int;         // Const only: reflection of the literal expression
  // Activity analysis flags (stamped by mark_activity):
  //   vself = this node depends on a differentiated input ("varied");
  //   va/vb = operand a/b is varied. A non-varied value has a statically-zero
  //   derivative, so its derivative work is elided.
  //   nself = this node's primal value is actually needed (read by some emitted
  //   derivative, transitively). A primal that feeds only the function's value
  //   -- never a derivative -- is dead in a pure derivative/gradient and is not
  //   emitted (so e.g. a `exp(x)` term drops out of d/dy entirely).
  // Defaults are "everything active/needed" (behaviour before analysis).
  bool vself = true, va = true, vb = true, nself = true;
};

// Which operands a node reads (Input/Const read none; binary ops read a and b;
// everything else reads only a).
consteval bool op_has_a(OpKind op) {
  return op != OpKind::Input && op != OpKind::Const;
}
consteval bool op_has_b(OpKind op) {
  return op == OpKind::Add || op == OpKind::Sub ||
         op == OpKind::Mul || op == OpKind::Div ||
         op == OpKind::Matmul;
}

namespace detail {

struct Ctx {
  std::vector<Node> nodes;
  std::vector<info> envDecl;      // decl -> slot environment
  std::vector<std::size_t> envSlot;
};

// Resolve a referenced variable to its SSA slot by identifier name. We match by
// name (unique within a straight-line body) rather than reflection identity
// because parameters_of yields ReflectionKind::Parameter while a decl_ref's
// declaration_of yields ReflectionKind::Declaration — different kinds that never
// compare equal. Last-wins so a local shadowing an outer name resolves to the
// most recent binding.
consteval std::size_t findSlot(const Ctx &c, std::string_view name) {
  for (std::size_t i = c.envDecl.size(); i-- > 0;)
    if (m::identifier_of(c.envDecl[i]) == name)
      return c.envSlot[i];
  return static_cast<std::size_t>(-1);
}

consteval OpKind binOp(m::operators op) {
  if (op == m::operators::op_plus)  return OpKind::Add;
  if (op == m::operators::op_minus) return OpKind::Sub;
  if (op == m::operators::op_star)  return OpKind::Mul;
  if (op == m::operators::op_slash) return OpKind::Div;
  return OpKind::Add;  // unsupported binary op (v1)
}

consteval OpKind callOp(std::string_view name) {
  // named arithmetic (used when operands are class types, e.g. tensors, where
  // `a + b` would be an operator-call; named ops keep the reflected AST simple)
  if (name == "add") return OpKind::Add;
  if (name == "sub") return OpKind::Sub;
  if (name == "mul") return OpKind::Mul;
  if (name == "div") return OpKind::Div;
  if (name == "sin")  return OpKind::Sin;
  if (name == "cos")  return OpKind::Cos;
  if (name == "exp")  return OpKind::Exp;
  if (name == "log")  return OpKind::Log;
  if (name == "sqrt") return OpKind::Sqrt;
  if (name == "matmul")    return OpKind::Matmul;
  if (name == "transpose") return OpKind::Transpose;
  if (name == "sum")       return OpKind::Sum;
  if (name == "relu")      return OpKind::Relu;
  return OpKind::Sin;  // unsupported call
}

// Peel "transparent" wrapper nodes so we reach the real subexpression:
//   - implicit casts (lvalue-to-rvalue, NoOp, conversions), and
//   - single-argument copy/move constructions, which wrap e.g. `return v;` for
//     class-typed (tensor) functions. A multi-arg construct is a real object
//     construction and is left intact.
consteval info stripCasts(info e) {
  while (m::is_expression(e)) {
    if (m::is_cast(e))
      e = m::operands_of(e).front();
    else if (m::is_construct(e) && m::operands_of(e).size() == 1)
      e = m::operands_of(e).front();
    else
      break;
  }
  return e;
}

// Lower an expression to SSA nodes, returning its result slot.
consteval std::size_t lower(Ctx &c, info e) {
  e = stripCasts(e);

  if (m::is_variable_reference(e))
    return findSlot(c, m::identifier_of(m::declaration_of(e)));

  if (m::is_literal(e)) {
    std::size_t s = c.nodes.size();
    // Reduce the literal to a value reflection (constant_of); it is spliced by
    // the existing P2996 value splice at codegen -- no expression splicing.
    c.nodes.push_back(Node{OpKind::Const, s, 0, 0, m::constant_of(e)});
    return s;
  }

  if (m::is_unary_operator(e)) {
    std::size_t a = lower(c, m::operands_of(e).front());
    m::operators op = m::expression_operator_of(e);
    if (op == m::operators::op_plus)
      return a;  // unary plus is a no-op
    if (op == m::operators::op_minus) {
      std::size_t s = c.nodes.size();
      c.nodes.push_back(Node{OpKind::Neg, s, a, 0});
      return s;
    }
    throw "Unsupported unary operator"; 
  }

  if (m::is_binary_operator(e)) {
    auto ops = m::operands_of(e);
    std::size_t a = lower(c, ops[0]);
    std::size_t b = lower(c, ops[1]);
    std::size_t s = c.nodes.size();
    c.nodes.push_back(Node{binOp(m::expression_operator_of(e)), s, a, b});
    return s;
  }

  if (m::is_function_call(e)) {
    // operands_of(call) = [callee, arg0, arg1, ...].
    auto ops = m::operands_of(e);
    OpKind op = callOp(m::identifier_of(m::callee_of(e)));
    if (op_has_b(op)) {                    // binary call, e.g. matmul(a, b)
      std::size_t a = lower(c, ops[1]);
      std::size_t b = lower(c, ops[2]);
      std::size_t s = c.nodes.size();
      c.nodes.push_back(Node{op, s, a, b});
      return s;
    }
    std::size_t a = lower(c, ops[ops.size() - 1]);   // unary call
    std::size_t s = c.nodes.size();
    c.nodes.push_back(Node{op, s, a, 0});
    return s;
  }

  // Unsupported: emit a constant placeholder from its value.
  std::size_t s = c.nodes.size();
  c.nodes.push_back(Node{OpKind::Const, s, 0, 0, m::constant_of(e)});
  return s;
}

}  // namespace detail

// Build the SSA node list for a reflected function. The last node is the root
// (an Output node forwarding the return value).
template <info Fn>
consteval std::vector<Node> build_nodes() {
  detail::Ctx c;
  for (info p : m::parameters_of(Fn)) {
    std::size_t s = c.nodes.size();
    c.nodes.push_back(Node{OpKind::Input, s, 0, 0});
    c.envDecl.push_back(p);
    c.envSlot.push_back(s);
  }

  std::size_t root = 0;
  for (info s : m::statements_of(m::body_of(Fn))) {
    if (m::is_declaration_statement(s)) {
      info v = m::declared_variable_of(s);
      std::size_t slot = detail::lower(c, m::initializer_of(v));
      c.envDecl.push_back(v);
      c.envSlot.push_back(slot);
    } else if (m::is_return_statement(s)) {
      root = detail::lower(c, m::return_value_of(s));
    }
  }

  std::size_t s = c.nodes.size();
  c.nodes.push_back(Node{OpKind::Output, s, root, 0});
  return c.nodes;
}

// Activity analysis: stamp each node's varied flags. A node is "varied" if it
// (transitively) depends on a differentiated input. `wrt >= 0` marks only that
// one input active (forward mode, single directional derivative); `wrt < 0`
// marks all inputs active (reverse mode, full gradient). Nodes are in
// topological order, so operands precede their users and one forward pass
// suffices. Derivative work for non-varied values is a statically-zero term and
// is elided by the codegen, which removes e.g. the `x * 0` from a constant
// factor's derivative.
// Ops whose derivative rule reads the primal value of an operand / of itself.
consteval bool deriv_reads_operand_vals(OpKind op) {  // reads val[a] (and val[b])
  return op == OpKind::Mul || op == OpKind::Div ||
         op == OpKind::Sin || op == OpKind::Cos || op == OpKind::Log;
}
consteval bool deriv_reads_self_val(OpKind op) {       // reads val[self]
  return op == OpKind::Exp || op == OpKind::Sqrt;
}

// `active_mask` bit i set => input i is differentiated w.r.t. (~0ull = all).
consteval void mark_activity(std::vector<Node> &ns, unsigned long long active_mask) {
  const std::size_t N = ns.size();

  // 1. varied: forward reachability from the differentiated input(s).
  std::vector<char> v(N, 0);
  for (Node &n : ns) {
    bool varied;
    if (n.op == OpKind::Input)
      varied = (active_mask >> n.self) & 1ull;
    else if (n.op == OpKind::Const)
      varied = false;
    else
      varied = (op_has_a(n.op) && v[n.a]) || (op_has_b(n.op) && v[n.b]);
    v[n.self] = varied ? 1 : 0;
    n.vself = varied;
    n.va = op_has_a(n.op) && v[n.a];
    n.vb = op_has_b(n.op) && v[n.b];
  }

  // 2. needed: which primals are actually read by an emitted derivative. Seed
  // from the vals each active (varied) node's derivative rule reads, then
  // propagate backward through primal dependencies (to compute val[i] you need
  // its operands' vals). Primals that feed only the function value -- never a
  // derivative -- stay unneeded and are not emitted.
  std::vector<char> need(N, 0);
  for (Node &n : ns) {
    if (!n.vself)
      continue;
    if (deriv_reads_operand_vals(n.op)) {
      if (op_has_a(n.op)) need[n.a] = 1;
      if (op_has_b(n.op)) need[n.b] = 1;
    }
    if (deriv_reads_self_val(n.op))
      need[n.self] = 1;
  }
  for (std::size_t i = N; i-- > 0;) {   // backward: needed val pulls in operands
    if (!need[i])
      continue;
    const Node &n = ns[i];
    if (op_has_a(n.op)) need[n.a] = 1;
    if (op_has_b(n.op)) need[n.b] = 1;
  }
  for (Node &n : ns)
    n.nself = need[n.self];
}

// build_nodes + activity analysis, in topological order.
template <info Fn, unsigned long long ActiveMask>
consteval std::vector<Node> build_marked_nodes() {
  std::vector<Node> ns = build_nodes<Fn>();
  mark_activity(ns, ActiveMask);
  return ns;
}

// Same, reversed (for the adjoint sweep); marks are preserved.
template <info Fn, unsigned long long ActiveMask>
consteval std::vector<Node> build_marked_nodes_reversed() {
  std::vector<Node> fwd = build_marked_nodes<Fn, ActiveMask>();
  std::vector<Node> rev;
  rev.reserve(fwd.size());
  for (std::size_t i = fwd.size(); i-- > 0;)
    rev.push_back(fwd[i]);
  return rev;
}

// Forward-mode directional derivative of `Fn` w.r.t. input index `Wrt`,
// evaluated at `args`. Compiles to inlined arithmetic (zero-cost).
template <info Fn, std::size_t Wrt, typename T = double, typename... Args>
constexpr T forward_derivative(Args... args) {
  static constexpr auto nodes = std::define_static_array(build_marked_nodes<Fn, (1ull << Wrt)>());
  constexpr std::size_t N = nodes.size();

  const T in[] = { static_cast<T>(args)... };
  T val[N];
  T tang[N];

  template for (constexpr auto n : nodes) {
    // Primal (only where the value is actually read by a derivative).
    if constexpr (n.nself) {
      if constexpr (n.op == OpKind::Input)       val[n.self] = in[n.self];
      else if constexpr (n.op == OpKind::Const)  val[n.self] = static_cast<T>([: n.leaf :]);
      else if constexpr (n.op == OpKind::Output) val[n.self] = val[n.a];
      else if constexpr (n.op == OpKind::Add)    val[n.self] = val[n.a] + val[n.b];
      else if constexpr (n.op == OpKind::Sub)    val[n.self] = val[n.a] - val[n.b];
      else if constexpr (n.op == OpKind::Mul)    val[n.self] = val[n.a] * val[n.b];
      else if constexpr (n.op == OpKind::Div)    val[n.self] = val[n.a] / val[n.b];
      else if constexpr (n.op == OpKind::Neg)    val[n.self] = -val[n.a];
      else if constexpr (n.op == OpKind::Sin)    val[n.self] = std::sin(val[n.a]);
      else if constexpr (n.op == OpKind::Cos)    val[n.self] = std::cos(val[n.a]);
      else if constexpr (n.op == OpKind::Exp)    val[n.self] = std::exp(val[n.a]);
      else if constexpr (n.op == OpKind::Log)    val[n.self] = std::log(val[n.a]);
      else if constexpr (n.op == OpKind::Sqrt)   val[n.self] = std::sqrt(val[n.a]);
    }

    // Tangent (activity-gated: a non-varied operand contributes a zero term,
    // which is dropped instead of emitted as `... * 0`).
    if constexpr (n.op == OpKind::Input)       tang[n.self] = (n.self == Wrt) ? T{1} : T{0};
    else if constexpr (!n.vself)               tang[n.self] = T{0};  // not varied
    else if constexpr (n.op == OpKind::Output) tang[n.self] = tang[n.a];
    else if constexpr (n.op == OpKind::Neg)    tang[n.self] = -tang[n.a];
    else if constexpr (n.op == OpKind::Sin)    tang[n.self] = std::cos(val[n.a]) * tang[n.a];
    else if constexpr (n.op == OpKind::Cos)    tang[n.self] = -std::sin(val[n.a]) * tang[n.a];
    else if constexpr (n.op == OpKind::Exp)    tang[n.self] = val[n.self] * tang[n.a];
    else if constexpr (n.op == OpKind::Log)    tang[n.self] = tang[n.a] / val[n.a];
    else if constexpr (n.op == OpKind::Sqrt)   tang[n.self] = tang[n.a] / (T{2} * val[n.self]);
    else if constexpr (n.op == OpKind::Add) {
      if constexpr (n.va && n.vb) tang[n.self] = tang[n.a] + tang[n.b];
      else if constexpr (n.va)    tang[n.self] = tang[n.a];
      else                        tang[n.self] = tang[n.b];
    } else if constexpr (n.op == OpKind::Sub) {
      if constexpr (n.va && n.vb) tang[n.self] = tang[n.a] - tang[n.b];
      else if constexpr (n.va)    tang[n.self] = tang[n.a];
      else                        tang[n.self] = -tang[n.b];
    } else if constexpr (n.op == OpKind::Mul) {
      if constexpr (n.va && n.vb) tang[n.self] = tang[n.a] * val[n.b] + val[n.a] * tang[n.b];
      else if constexpr (n.va)    tang[n.self] = tang[n.a] * val[n.b];
      else                        tang[n.self] = val[n.a] * tang[n.b];
    } else if constexpr (n.op == OpKind::Div) {
      if constexpr (n.va && n.vb)
        tang[n.self] = (tang[n.a] * val[n.b] - val[n.a] * tang[n.b]) / (val[n.b] * val[n.b]);
      else if constexpr (n.va)    tang[n.self] = tang[n.a] / val[n.b];
      else                        tang[n.self] = -val[n.a] * tang[n.b] / (val[n.b] * val[n.b]);
    } else {
      tang[n.self] = T{0};  // Const and any unhandled op
    }
  }

  return tang[N - 1];
}

namespace detail {
template <info Fn, typename T, typename... Args, std::size_t... I>
constexpr std::array<T, sizeof...(I)> grad_impl(std::index_sequence<I...>,
                                                Args... args) {
  return { forward_derivative<Fn, I, T>(args...)... };
}
}  // namespace detail

// Full gradient via forward mode: one forward pass per input (P passes).
template <info Fn, typename T = double, typename... Args>
constexpr std::array<T, sizeof...(Args)> gradient_of(Args... args) {
  return detail::grad_impl<Fn, T>(std::index_sequence_for<Args...>{}, args...);
}

// Full gradient via reverse mode: one primal sweep + one adjoint sweep over the
// reversed DAG, computing every partial in a single pass. This is the efficient
// path for scalar-output, many-input functions (the "autograd" case).
template <info Fn, typename T = double, typename... Args>
constexpr std::array<T, sizeof...(Args)> gradient_reverse(Args... args) {
  static constexpr auto nodes = std::define_static_array(build_marked_nodes<Fn, ~0ull>());
  static constexpr auto rnodes = std::define_static_array(build_marked_nodes_reversed<Fn, ~0ull>());
  constexpr std::size_t N = nodes.size();
  constexpr std::size_t P = sizeof...(Args);

  const T in[] = { static_cast<T>(args)... };
  T val[N];
  T adj[N] = {};   // adjoints, accumulated; zero-initialized

  // Forward (primal) sweep: compute the values the adjoint rules will read.
  template for (constexpr auto n : nodes) {
    if constexpr (n.nself) {
      if constexpr (n.op == OpKind::Input)       val[n.self] = in[n.self];
      else if constexpr (n.op == OpKind::Const)  val[n.self] = static_cast<T>([: n.leaf :]);
      else if constexpr (n.op == OpKind::Output) val[n.self] = val[n.a];
      else if constexpr (n.op == OpKind::Add)    val[n.self] = val[n.a] + val[n.b];
      else if constexpr (n.op == OpKind::Sub)    val[n.self] = val[n.a] - val[n.b];
      else if constexpr (n.op == OpKind::Mul)    val[n.self] = val[n.a] * val[n.b];
      else if constexpr (n.op == OpKind::Div)    val[n.self] = val[n.a] / val[n.b];
      else if constexpr (n.op == OpKind::Neg)    val[n.self] = -val[n.a];
      else if constexpr (n.op == OpKind::Sin)    val[n.self] = std::sin(val[n.a]);
      else if constexpr (n.op == OpKind::Cos)    val[n.self] = std::cos(val[n.a]);
      else if constexpr (n.op == OpKind::Exp)    val[n.self] = std::exp(val[n.a]);
      else if constexpr (n.op == OpKind::Log)    val[n.self] = std::log(val[n.a]);
      else if constexpr (n.op == OpKind::Sqrt)   val[n.self] = std::sqrt(val[n.a]);
    }
  }

  // Seed the output adjoint, then sweep the DAG in reverse, pushing each node's
  // adjoint to its operands via the local VJP (accumulating with +=).
  adj[N - 1] = T{1};
  template for (constexpr auto n : rnodes) {
    // Push this node's adjoint to each operand, but only to *varied* operands
    // (a non-varied operand leads to no differentiated input, so the update is
    // a wasted `+= ... * 0` into a dead slot).
    if constexpr (n.op == OpKind::Output) {
      if constexpr (n.va) adj[n.a] += adj[n.self];
    } else if constexpr (n.op == OpKind::Add) {
      if constexpr (n.va) adj[n.a] += adj[n.self];
      if constexpr (n.vb) adj[n.b] += adj[n.self];
    } else if constexpr (n.op == OpKind::Sub) {
      if constexpr (n.va) adj[n.a] += adj[n.self];
      if constexpr (n.vb) adj[n.b] -= adj[n.self];
    } else if constexpr (n.op == OpKind::Mul) {
      if constexpr (n.va) adj[n.a] += adj[n.self] * val[n.b];
      if constexpr (n.vb) adj[n.b] += adj[n.self] * val[n.a];
    } else if constexpr (n.op == OpKind::Div) {
      if constexpr (n.va) adj[n.a] += adj[n.self] / val[n.b];
      if constexpr (n.vb) adj[n.b] -= adj[n.self] * val[n.a] / (val[n.b] * val[n.b]);
    } else if constexpr (n.op == OpKind::Neg) {
      if constexpr (n.va) adj[n.a] -= adj[n.self];
    } else if constexpr (n.op == OpKind::Sin) {
      if constexpr (n.va) adj[n.a] += adj[n.self] * std::cos(val[n.a]);
    } else if constexpr (n.op == OpKind::Cos) {
      if constexpr (n.va) adj[n.a] += -adj[n.self] * std::sin(val[n.a]);
    } else if constexpr (n.op == OpKind::Exp) {
      if constexpr (n.va) adj[n.a] += adj[n.self] * val[n.self];
    } else if constexpr (n.op == OpKind::Log) {
      if constexpr (n.va) adj[n.a] += adj[n.self] / val[n.a];
    } else if constexpr (n.op == OpKind::Sqrt) {
      if constexpr (n.va) adj[n.a] += adj[n.self] / (T{2} * val[n.self]);
    }
    // Input / Const have no operands: nothing to propagate.
  }

  // Input k's accumulated adjoint is the k-th partial derivative.
  std::array<T, P> g{};
  for (std::size_t i = 0; i < P; ++i)
    g[i] = adj[i];
  return g;
}

}  // namespace ad

#endif  // REFLECT_DEMO_AUTOGRAD_H
