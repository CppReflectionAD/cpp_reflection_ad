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
//   - partial_derivative<^^f, Wrt...>(args...) - Calculate the partial_derivative of `f`, with respect to each index in `Wrt...`
//        e.g. partial_derivative<^^f, 0, 0, 1>(args...) will differentiate `f` twice w.r.t the 0th argument, and once w.r.t. the first.
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
// sin/cos/exp/log/sqrt/erfc. The core is rule-driven so more ops / tensors are
// additive. Calls to user-defined helpers are inlined (the callee's body is
// reflected into the same DAG), so ordinary functions compose; the callee must
// itself be straight-line and non-recursive (free functions only).

#ifndef REFLECT_DEMO_AUTOGRAD_H
#define REFLECT_DEMO_AUTOGRAD_H

#include <meta>
#include <vector>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string_view>
#include <utility>

namespace ad {
using std::meta::info;
namespace m = std::meta;

using namespace std::numbers;
constexpr double two_over_root_pi = 2. * std::numbers::inv_sqrtpi_v<double>;

// ---------------------------------------------------------------------------
// IR
// ---------------------------------------------------------------------------
enum class OpKind {
  Input, Const, Output,
  Add, Sub, Mul, Neg, Inv,
  Sin, Cos, Exp, Log, Sqrt, Erfc,
  // Tensor ops (recognised as named calls; VJPs live in the tensor engine).
  Matmul, Transpose, Sum, Relu,
};

// A call is a *primitive* (has a built-in VJP) iff its callee is registered
// here; anything else is a user helper and gets inlined. The key is the callee
// reflection, not its name, so a same-named function in another namespace -- or
// a different overload -- is a different key and does not collide. Register a
// vocabulary with:
//   template <> struct ad::primitive<^^nn::relu> {
//     static constexpr ad::OpKind op = ad::OpKind::Relu;
//   };
template <info F> struct primitive;   // declared, never defined

// `^^f` is ill-formed when `f` names an overload set, so a specific overload is
// selected by signature:
//   template <> struct ad::primitive<ad::overload_of(^^nn, "sum", ^^Tensor(const Tensor &))>
consteval info overload_of(info scope, std::string_view name, info fnType) {
  for (info mem : m::members_of(scope, m::access_context::current()))
    if (m::has_identifier(mem) && m::identifier_of(mem) == name &&
        m::type_of(mem) == fnType)
      return mem;
  throw "reflection AD: no overload with that signature in this scope";
}

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
         op == OpKind::Mul || op == OpKind::Matmul;
}

namespace detail {

struct Ctx {
  std::vector<Node> nodes;
  std::vector<info> envDecl;      // decl -> slot environment
  std::vector<std::size_t> envSlot;
  std::vector<info> callStack;    // callees currently being inlined (cycle guard)
};

// Forward decls: lower / lower_body / inline_call are mutually recursive (a call
// site inlines its callee's body, which may contain further calls).
consteval std::size_t lower(Ctx &c, info e);
consteval std::size_t lower_body(Ctx &c, info body);
consteval std::size_t inline_call(Ctx &c, info call);

// Resolve a referenced variable to its SSA slot by identifier name. We match by
// name (unique within a straight-line body) rather than reflection identity
// because parameters_of yields ReflectionKind::Parameter while a decl_ref's
// declaration_of yields ReflectionKind::Declaration — different kinds that never
// compare equal. Last-wins so a local shadowing an outer name resolves to the
// most recent binding. An unresolved name (e.g. a global) has no slot; erroring
// beats returning a sentinel that would index the node array out of bounds.
consteval std::size_t findSlot(const Ctx &c, std::string_view name) {
  for (std::size_t i = c.envDecl.size(); i-- > 0;)
    if (m::identifier_of(c.envDecl[i]) == name)
      return c.envSlot[i];
  throw "reflection AD: name is not a parameter or local of the reflected body";
}

consteval OpKind binOp(m::operators op) {
  if (op == m::operators::op_plus)  return OpKind::Add;
  if (op == m::operators::op_minus) return OpKind::Sub;
  if (op == m::operators::op_star)  return OpKind::Mul;
  if (op == m::operators::op_slash) throw "Should not get here. `/` is a special case.";
  throw "Unsupported binary operator";
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

// The callee of the single call in a one-line probe function's body.
consteval info probe_callee(info fn) {
  for (info s : m::statements_of(m::body_of(fn)))
    if (m::is_return_statement(s))
      return m::callee_of(stripCasts(m::return_value_of(s)));
  throw "reflection AD: probe function has no call to reflect";
}

// `^^std::sin` is ill-formed -- it names an overload set, not a function -- so
// the canonical reflection of each std overload we differentiate is recovered
// from our own call to it. Matching a user call against these is exact: a
// user-defined `sin` is a different reflection and is inlined instead. One
// instantiation per real floating-point overload; an integral argument selects
// <cmath>'s integral template, which is not covered.
template <class T> inline T p_sin (T x) { return std::sin(x); }
template <class T> inline T p_cos (T x) { return std::cos(x); }
template <class T> inline T p_exp (T x) { return std::exp(x); }
template <class T> inline T p_log (T x) { return std::log(x); }
template <class T> inline T p_sqrt(T x) { return std::sqrt(x); }
template <class T> inline T p_erfc(T x) { return std::erfc(x); }

struct Prim { bool found; OpKind op; };

// Does `callee` match the float / double / long double instantiations of a
// probe template?
consteval bool is_std_fn(info callee, info pf, info pd, info pl) {
  return callee == probe_callee(pf) || callee == probe_callee(pd) ||
         callee == probe_callee(pl);
}

// Is `callee` a primitive? Checks the built-in std math set, then the
// user-extensible ad::primitive registry.
consteval Prim find_primitive(info callee) {
  if (is_std_fn(callee, ^^p_sin<float>,  ^^p_sin<double>,  ^^p_sin<long double>))
    return {true, OpKind::Sin};
  if (is_std_fn(callee, ^^p_cos<float>,  ^^p_cos<double>,  ^^p_cos<long double>))
    return {true, OpKind::Cos};
  if (is_std_fn(callee, ^^p_exp<float>,  ^^p_exp<double>,  ^^p_exp<long double>))
    return {true, OpKind::Exp};
  if (is_std_fn(callee, ^^p_log<float>,  ^^p_log<double>,  ^^p_log<long double>))
    return {true, OpKind::Log};
  if (is_std_fn(callee, ^^p_sqrt<float>, ^^p_sqrt<double>, ^^p_sqrt<long double>))
    return {true, OpKind::Sqrt};
  if (is_std_fn(callee, ^^p_erfc<float>, ^^p_erfc<double>, ^^p_erfc<long double>))
    return {true, OpKind::Erfc};

  info spec = m::substitute(^^primitive, {m::reflect_constant(callee)});
  if (m::is_complete_type(spec))
    for (info mem : m::members_of(spec, m::access_context::current()))
      if (m::has_identifier(mem) && m::identifier_of(mem) == "op")
        return {true, m::extract<OpKind>(mem)};
  return {false, OpKind::Input};
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
    // `a / b` becomes `a * inv(b)`.
    if (m::expression_operator_of(e) == m::operators::op_slash) {
      std::size_t inv = c.nodes.size();
      c.nodes.push_back(Node{OpKind::Inv, inv, b, 0});
      std::size_t s = c.nodes.size();
      c.nodes.push_back(Node{OpKind::Mul, s, a, inv});
      return s;
    }
    std::size_t s = c.nodes.size();
    c.nodes.push_back(Node{binOp(m::expression_operator_of(e)), s, a, b});
    return s;
  }

  if (m::is_function_call(e)) {
    // operands_of(call) = [callee, arg0, arg1, ...].
    auto ops = m::operands_of(e);
    Prim p = find_primitive(m::callee_of(e));
    if (!p.found)                         // user helper -> inline its body
      return inline_call(c, e);
    OpKind op = p.op;
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

// Lower the statements of a straight-line body into `c`, resolving names against
// the current environment; returns the slot of the (single) return value. Shared
// by build_nodes (top-level function) and inline_call (inlined callee).
// Anything outside that shape (control flow, assignment, a second return, no
// return at all) is rejected: skipping it would silently build a DAG that does
// not match the source, i.e. a wrong derivative with no diagnostic.
consteval std::size_t lower_body(Ctx &c, info body) {
  std::size_t root = 0;
  bool sawReturn = false;
  for (info s : m::statements_of(body)) {
    if (m::is_declaration_statement(s)) {
      info v = m::declared_variable_of(s);
      std::size_t slot = lower(c, m::initializer_of(v));
      c.envDecl.push_back(v);
      c.envSlot.push_back(slot);
    } else if (m::is_return_statement(s)) {
      if (sawReturn)
        throw "reflection AD: multiple return statements (straight-line bodies only)";
      root = lower(c, m::return_value_of(s));
      sawReturn = true;
    } else {
      throw "reflection AD: unsupported statement; straight-line bodies only "
            "(`T v = expr;` declarations and one `return expr;`)";
    }
  }
  if (!sawReturn)
    throw "reflection AD: body has no return statement";
  return root;
}

// Inline a call to a user-defined helper: lower its arguments in the caller
// scope, bind the callee's parameters to those argument slots, and lower the
// callee's body into the SAME DAG. The result is the callee's return slot, so
// the whole helper expands to primitive nodes with no new IR kinds -- activity
// analysis and codegen are unaffected, and arguments are evaluated once (a param
// used N times reuses its one slot).
consteval std::size_t inline_call(Ctx &c, info call) {
  auto ops = m::operands_of(call);          // [callee, arg0, arg1, ...]
  info callee = m::callee_of(call);

  // Cycle guard: recursive helpers are out of scope (would not terminate here).
  for (info f : c.callStack)
    if (f == callee)
      throw "reflection AD: cannot inline a recursive function call";

  // A helper with no reflectable body (declaration only) cannot be inlined; a
  // silent constant would give a wrong derivative, so reject it explicitly.
  info body = m::body_of(callee);
  if (m::statements_of(body).empty())
    throw "reflection AD: cannot inline a call whose callee has no visible body";

  // (a) lower argument expressions in the CALLER scope -> slots.
  std::vector<std::size_t> argSlots;
  for (std::size_t i = 1; i < ops.size(); ++i)
    argSlots.push_back(lower(c, ops[i]));

  // (b) open a lexical scope; bind callee parameters to the argument slots.
  std::size_t mark = c.envDecl.size();
  auto params = m::parameters_of(callee);
  for (std::size_t i = 0; i < params.size(); ++i) {
    c.envDecl.push_back(params[i]);
    c.envSlot.push_back(argSlots[i]);
  }

  // (c) lower the callee body into the same DAG (nested calls inline via lower).
  c.callStack.push_back(callee);
  std::size_t ret = lower_body(c, body);
  c.callStack.pop_back();

  // (d) close the scope: drop callee params + locals, restoring the caller env.
  c.envDecl.resize(mark);
  c.envSlot.resize(mark);
  return ret;
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

  std::size_t root = detail::lower_body(c, m::body_of(Fn));

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
  return op == OpKind::Mul ||
         op == OpKind::Sin || op == OpKind::Cos || op == OpKind::Log ||
         op == OpKind::Erfc;
}

consteval bool deriv_reads_self_val(OpKind op) {       // reads val[self]
  return op == OpKind::Exp || op == OpKind::Sqrt || op == OpKind::Inv;
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

// ---------------------------------------------------------------------------
// Higher-order derivatives by recursive DAG differentiation.
// ---------------------------------------------------------------------------
namespace detail {

// Raw append: always pushes a new node and returns its slot.
consteval std::size_t emit_raw(std::vector<Node> &out, OpKind op,
                               std::size_t a, std::size_t b, info leaf = ^^int) {
  std::size_t s = out.size();
  out.push_back(Node{op, s, a, b, leaf});
  return s;
}

// If we have an existing node in the DAG with the same operation and operands, we
// can reuse it instead of bloating the DAG with duplicate nodes.
consteval std::size_t emit_node(std::vector<Node> &out, OpKind op,
                                std::size_t a, std::size_t b, info leaf = ^^int) {
  for (const Node &n : out)
    if (n.op == op && n.a == a && n.b == b)
      return n.self;
  return emit_raw(out, op, a, b, leaf);
}

// A compile-time map from a constant scalar value to the slot of its (unique)
// Const node.
using ConstPool = std::vector<std::pair<double, std::size_t> >;

// Return the slot of the Const node holding `value`, creating it (and recording
// it in `pool`) on first request. `emit_node` caches based on operands, but `ConstPool`
// caches based on the leaf value.
consteval std::size_t ensure_const_node(std::vector<Node> &out,
                                        ConstPool &pool, double value) {
  for (const auto &[v, slot] : pool)
    if (v == value)
      return slot;
  std::size_t slot =
      emit_raw(out, OpKind::Const, 0, 0, m::reflect_constant(value));
  pool.push_back({value, slot});
  return slot;
}

// Build the derivative DAG of `src` w.r.t. input index `wrt`.
consteval std::vector<Node> differentiate(const std::vector<Node> &src,
                                          std::size_t wrt) {
  const std::size_t M = src.size();
  std::vector<Node> out;
  std::vector<std::size_t> tang(M, 0);   // tang[i] returns the slot in `out` that corresponds to the tangent of the ith node in `src`.
  std::vector<bool> varied(M, 0);        // varied[i] marks whether the tangent of ith node in 'src' is non-zero.

  // Copy every primal node except the Output (which is at the end of src).
  // Copying in order keeps each node at its original slot, so `src` Node operands are not invalidated.
  // We assume these will be reused in the calculate for the derivative, and if not they will be pruned
  // later.
  for (const Node &n : src)
    if (n.op != OpKind::Output)
      emit_raw(out, n.op, n.a, n.b, n.leaf);

  ConstPool constPool;

  // Create a tangent node for each `src` node in order.
  for (const Node &n : src) {
    const std::size_t i = n.self;
    // `va`/`vb` correspond to whether operand a/b has a nonzero tangent
    const bool va = op_has_a(n.op) && varied[n.a];
    const bool vb = op_has_b(n.op) && varied[n.b];
    switch (n.op) {
      case OpKind::Input:
        varied[i] = (i == wrt);
        if (varied[i]) tang[i] = ensure_const_node(out, constPool, 1.0);
        break;
      case OpKind::Const:
        varied[i] = false;
        break;
      case OpKind::Output:                  // forwards the return value's tangent
        varied[i] = va;
        if (va) tang[i] = tang[n.a];
        break;
      case OpKind::Add:
        varied[i] = va || vb;
        if (va && vb) tang[i] = emit_node(out, OpKind::Add, tang[n.a], tang[n.b]);
        else if (va)  tang[i] = tang[n.a];
        else if (vb)  tang[i] = tang[n.b];
        break;
      case OpKind::Sub:
        varied[i] = va || vb;
        if (va && vb) tang[i] = emit_node(out, OpKind::Sub, tang[n.a], tang[n.b]);
        else if (va)  tang[i] = tang[n.a];
        else if (vb)  tang[i] = emit_node(out, OpKind::Neg, tang[n.b], 0);
        break;
      case OpKind::Mul:                     // d(ab) = da*b + a*db
        varied[i] = va || vb;
        if (va && vb) {
          // product rule
          std::size_t l = emit_node(out, OpKind::Mul, tang[n.a], n.b);
          std::size_t r = emit_node(out, OpKind::Mul, n.a, tang[n.b]);
          tang[i] = emit_node(out, OpKind::Add, l, r);
        } else if (va) tang[i] = emit_node(out, OpKind::Mul, tang[n.a], n.b);
        else if (vb)   tang[i] = emit_node(out, OpKind::Mul, n.a, tang[n.b]);
        break;
      case OpKind::Inv:                     // d(1/a) = -(1/a)*(1/a) * da
        varied[i] = va;
        if (va) {
          std::size_t inv_a_squared = emit_node(out, OpKind::Mul, i, i);
          std::size_t p = emit_node(out, OpKind::Mul, inv_a_squared, tang[n.a]);
          tang[i] = emit_node(out, OpKind::Neg, p, 0);
        }
        break;
      case OpKind::Neg:
        varied[i] = va;
        if (va) tang[i] = emit_node(out, OpKind::Neg, tang[n.a], 0);
        break;
      case OpKind::Sin:                     // cos(a) * da
        varied[i] = va;
        if (va) {
          std::size_t c = emit_node(out, OpKind::Cos, n.a, 0);
          tang[i] = emit_node(out, OpKind::Mul, c, tang[n.a]);
        }
        break;
      case OpKind::Cos:                     // -sin(a) * da
        varied[i] = va;
        if (va) {
          std::size_t s = emit_node(out, OpKind::Sin, n.a, 0);
          std::size_t p = emit_node(out, OpKind::Mul, s, tang[n.a]);
          tang[i] = emit_node(out, OpKind::Neg, p, 0);
        }
        break;
      case OpKind::Exp:                     // exp(a) * da
        varied[i] = va;
        // slot i is `exp(a)`
        if (va) tang[i] = emit_node(out, OpKind::Mul, i, tang[n.a]);
        break;
      case OpKind::Log:                     // da * (1/a)
        varied[i] = va;
        if (va) {
          std::size_t inv_a = emit_node(out, OpKind::Inv, n.a, 0);
          tang[i] = emit_node(out, OpKind::Mul, tang[n.a], inv_a);
        }
        break;
      case OpKind::Sqrt:                    // da * 1/(2*sqrt(a)) = da * 1/(2 * self)
        varied[i] = va;
        if (va) {
          std::size_t two = ensure_const_node(out, constPool, 2.0);
          // slot i is `sqrt(a)`
          std::size_t two_sqrt_a  = emit_node(out, OpKind::Mul, two, i);
          std::size_t inv_two_sqrt_a = emit_node(out, OpKind::Inv, two_sqrt_a, 0);
          tang[i] = emit_node(out, OpKind::Mul, tang[n.a], inv_two_sqrt_a);
        }
        break;
      case OpKind::Erfc:                    // -2/sqrt(pi) * exp(-a*a) * da
        varied[i] = va;
        if (va) {
          std::size_t k   = ensure_const_node(out, constPool, -two_over_root_pi);
          std::size_t a_squared  = emit_node(out, OpKind::Mul, n.a, n.a);
          std::size_t minus_a_squared = emit_node(out, OpKind::Neg, a_squared, 0);
          std::size_t exp_minus_a_squared   = emit_node(out, OpKind::Exp, minus_a_squared, 0);
          std::size_t ke  = emit_node(out, OpKind::Mul, k, exp_minus_a_squared);
          tang[i] = emit_node(out, OpKind::Mul, ke, tang[n.a]);
        }
        break;
      default:
        throw "Unhandled";
        break;
    }
  }

  // Create a new output node.
  const std::size_t srcOutputIndex = M - 1;  
  std::size_t srcOutputDerivative = varied[srcOutputIndex]
      ? tang[srcOutputIndex]
      : ensure_const_node(out, constPool, 0.0);
  emit_raw(out, OpKind::Output, srcOutputDerivative, 0);
  return out;
}

// Mark only nodes reachable from the Output as needed, so the evaluator 
// skips nodes that are not needed.
consteval void prune_reachable(std::vector<Node> &ns) {
  const std::size_t N = ns.size();
  std::vector<char> need(N, 0);
  need[N - 1] = 1;
  for (std::size_t k = N; k-- > 0;) {
    if (!need[k]) continue;
    const Node &n = ns[k];
    if (op_has_a(n.op)) need[n.a] = 1;
    if (op_has_b(n.op)) need[n.b] = 1;
  }
  for (Node &n : ns)
    n.nself = need[n.self];
}

}  // namespace detail

// Create nodes for a DAG corresponding to the partial derivative of `Fn`, differentiated with
// respect to each argument in `Wrts...`. Creates a DAG for the original `Fn`, and then recursively
// creates subsequent DAGs for each derivative using forward AD.
template <info Fn, std::size_t... Wrts>
consteval std::vector<Node> build_partial_nodes() {
  std::vector<Node> ns = build_nodes<Fn>();
  const std::size_t list[] = { Wrts... };
  for (std::size_t w : list)
    ns = detail::differentiate(ns, w);
  detail::prune_reachable(ns);
  return ns;
}

// Partial derivative of `Fn` with respect to each index in `Wrts...`, evaluated at `Arg...`.
template <info Fn, std::size_t... Wrts, typename... Args>
constexpr double partial_derivative(Args... args) {
  static constexpr auto nodes = std::define_static_array(build_partial_nodes<Fn, Wrts...>());
  constexpr std::size_t N = nodes.size();
  const double in[] = { static_cast<double>(args)... };
  double val[N];
  template for (constexpr auto n : nodes) {
    if constexpr (n.nself) {
      if constexpr (n.op == OpKind::Input)       val[n.self] = in[n.self];
      else if constexpr (n.op == OpKind::Const)  val[n.self] = static_cast<double>([: n.leaf :]);
      else if constexpr (n.op == OpKind::Output) val[n.self] = val[n.a];
      else if constexpr (n.op == OpKind::Add)    val[n.self] = val[n.a] + val[n.b];
      else if constexpr (n.op == OpKind::Sub)    val[n.self] = val[n.a] - val[n.b];
      else if constexpr (n.op == OpKind::Mul)    val[n.self] = val[n.a] * val[n.b];
      else if constexpr (n.op == OpKind::Neg)    val[n.self] = -val[n.a];
      else if constexpr (n.op == OpKind::Inv)    val[n.self] = 1.0 / val[n.a];
      else if constexpr (n.op == OpKind::Sin)    val[n.self] = std::sin(val[n.a]);
      else if constexpr (n.op == OpKind::Cos)    val[n.self] = std::cos(val[n.a]);
      else if constexpr (n.op == OpKind::Exp)    val[n.self] = std::exp(val[n.a]);
      else if constexpr (n.op == OpKind::Log)    val[n.self] = std::log(val[n.a]);
      else if constexpr (n.op == OpKind::Sqrt)   val[n.self] = std::sqrt(val[n.a]);
      else if constexpr (n.op == OpKind::Erfc)   val[n.self] = std::erfc(val[n.a]);
    }
  }
  return val[N - 1];
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
      else if constexpr (n.op == OpKind::Neg)    val[n.self] = -val[n.a];
      else if constexpr (n.op == OpKind::Inv)    val[n.self] = T{1} / val[n.a];
      else if constexpr (n.op == OpKind::Sin)    val[n.self] = std::sin(val[n.a]);
      else if constexpr (n.op == OpKind::Cos)    val[n.self] = std::cos(val[n.a]);
      else if constexpr (n.op == OpKind::Exp)    val[n.self] = std::exp(val[n.a]);
      else if constexpr (n.op == OpKind::Log)    val[n.self] = std::log(val[n.a]);
      else if constexpr (n.op == OpKind::Sqrt)   val[n.self] = std::sqrt(val[n.a]);
      else if constexpr (n.op == OpKind::Erfc)   val[n.self] = std::erfc(val[n.a]);
    }

    // Tangent (activity-gated: a non-varied operand contributes a zero term,
    // which is dropped instead of emitted as `... * 0`).
    if constexpr (n.op == OpKind::Input)       tang[n.self] = (n.self == Wrt) ? T{1} : T{0};
    else if constexpr (!n.vself)               tang[n.self] = T{0};  // not varied
    else if constexpr (n.op == OpKind::Output) tang[n.self] = tang[n.a];
    else if constexpr (n.op == OpKind::Neg)    tang[n.self] = -tang[n.a];
    else if constexpr (n.op == OpKind::Inv)    tang[n.self] = -val[n.self] * val[n.self] * tang[n.a];
    else if constexpr (n.op == OpKind::Sin)    tang[n.self] = std::cos(val[n.a]) * tang[n.a];
    else if constexpr (n.op == OpKind::Cos)    tang[n.self] = -std::sin(val[n.a]) * tang[n.a];
    else if constexpr (n.op == OpKind::Exp)    tang[n.self] = val[n.self] * tang[n.a];
    else if constexpr (n.op == OpKind::Log)    tang[n.self] = tang[n.a] / val[n.a];
    else if constexpr (n.op == OpKind::Sqrt)   tang[n.self] = tang[n.a] / (T{2} * val[n.self]);
    else if constexpr (n.op == OpKind::Erfc)   tang[n.self] = tang[n.a] * -1 * two_over_root_pi * (std::exp(-1 * (val[n.a] * val[n.a]))) ;
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
      else if constexpr (n.op == OpKind::Neg)    val[n.self] = -val[n.a];
      else if constexpr (n.op == OpKind::Inv)    val[n.self] = T{1} / val[n.a];
      else if constexpr (n.op == OpKind::Sin)    val[n.self] = std::sin(val[n.a]);
      else if constexpr (n.op == OpKind::Cos)    val[n.self] = std::cos(val[n.a]);
      else if constexpr (n.op == OpKind::Exp)    val[n.self] = std::exp(val[n.a]);
      else if constexpr (n.op == OpKind::Log)    val[n.self] = std::log(val[n.a]);
      else if constexpr (n.op == OpKind::Sqrt)   val[n.self] = std::sqrt(val[n.a]);
      else if constexpr (n.op == OpKind::Erfc)   val[n.self] = std::erfc(val[n.a]);
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
    } else if constexpr (n.op == OpKind::Inv) {
      if constexpr (n.va) adj[n.a] -= adj[n.self] * val[n.self] * val[n.self];
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
    } else if constexpr (n.op == OpKind::Erfc) {
      if constexpr (n.va) adj[n.a] += adj[n.self] * -1 * two_over_root_pi * (std::exp(-1 * (val[n.a] * val[n.a]))) ;
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
