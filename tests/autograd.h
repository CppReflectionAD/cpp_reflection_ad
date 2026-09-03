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
//
// Control flow so far means `c ? a : b`, the operators building `c`, and
// abs/max/min; `if` and loops are not supported. Branches are predicated, not
// eager: each node carries its branch's condition (Node::guard) and every
// sweep skips nodes whose guard is false, so `x > 0 ? sqrt(x) : 0.0` never
// calls sqrt at x < 0 -- which would leak NaN into the gradient via `0 / NaN`.
// The derivative is that of the branch taken: correct almost everywhere.

#ifndef REFLECT_DEMO_AUTOGRAD_H
#define REFLECT_DEMO_AUTOGRAD_H

#include <meta>
#include <vector>
#include <algorithm>
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
  Add, Sub, Mul, Div, Neg,
  Sin, Cos, Exp, Log, Sqrt, Erfc,
  // Conditions and their combinators. Piecewise constant, so never "varied";
  // valued 0 or 1 in the same array as everything else.
  Lt, Le, Gt, Ge, Eq, Ne, And, Or, Not,
  // Select is `c ? a : b`; Abs/Max/Min are the kinks.
  Select, Abs, Max, Min,
  // Tensor ops (recognised as named calls; VJPs live in the tensor engine).
  Matmul, Transpose, Sum, Relu,
};

// Node::guard value meaning "always run". A guard is a slot index and slot 0
// is a real node, so the sentinel must be a value no slot can take.
inline constexpr std::size_t UNGUARDED = static_cast<std::size_t>(-1);

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
  // Select only. After `leaf` so positional aggregate inits still hold.
  std::size_t cond = 0;
  // Condition gating this node; UNGUARDED means "always". Honoured by every
  // sweep, so an untaken branch is never evaluated.
  std::size_t guard = UNGUARDED;
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

// Which operands a node reads. The Select condition is kept out of a/b so
// activity analysis does not treat it as a differentiable operand.
consteval bool op_has_a(OpKind op) {
  return op != OpKind::Input && op != OpKind::Const;
}
consteval bool op_has_b(OpKind op) {
  return op == OpKind::Add || op == OpKind::Sub ||
         op == OpKind::Mul || op == OpKind::Div ||
         op == OpKind::Lt || op == OpKind::Le ||
         op == OpKind::Gt || op == OpKind::Ge ||
         op == OpKind::Eq || op == OpKind::Ne ||
         op == OpKind::And || op == OpKind::Or ||
         op == OpKind::Select || op == OpKind::Max || op == OpKind::Min ||
         op == OpKind::Matmul;
}
consteval bool op_has_cond(OpKind op) { return op == OpKind::Select; }

// Ops valued 0/1. Derivative identically zero, so never varied.
consteval bool op_is_boolean(OpKind op) {
  return op == OpKind::Lt || op == OpKind::Le ||
         op == OpKind::Gt || op == OpKind::Ge ||
         op == OpKind::Eq || op == OpKind::Ne ||
         op == OpKind::And || op == OpKind::Or || op == OpKind::Not;
}

namespace detail {

struct Ctx {
  std::vector<Node> nodes;
  std::vector<info> envDecl;      // decl -> slot environment
  std::vector<std::size_t> envSlot;
  std::vector<info> callStack;    // callees currently being inlined (cycle guard)
  // The guard in force while lowering. UNGUARDED at the top level; a ternary
  // narrows it for the duration of each branch.
  std::size_t curGuard = UNGUARDED;
};

// Append a node carrying the guard currently in force.
consteval std::size_t emit(Ctx &c, OpKind op, std::size_t a, std::size_t b = 0,
                           info leaf = ^^int, std::size_t cond = 0) {
  std::size_t s = c.nodes.size();
  c.nodes.push_back(Node{op, s, a, b, leaf, cond, c.curGuard});
  return s;
}

// Tighten the enclosing guard by also requiring `p`. Nothing to combine with
// at the top level, so a non-nested ternary emits no And node.
consteval std::size_t narrow_guard(Ctx &c, std::size_t outer, std::size_t p) {
  if (outer == UNGUARDED)
    return p;
  return emit(c, OpKind::And, outer, p);
}

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

// Erroring beats a plausible default: falling back to Add would turn `x < y`
// into `x + y` -- a wrong derivative with no diagnostic.
consteval OpKind binOp(m::operators op) {
  if (op == m::operators::op_plus)  return OpKind::Add;
  if (op == m::operators::op_minus) return OpKind::Sub;
  if (op == m::operators::op_star)  return OpKind::Mul;
  if (op == m::operators::op_slash) return OpKind::Div;
  if (op == m::operators::op_less)               return OpKind::Lt;
  if (op == m::operators::op_less_equals)        return OpKind::Le;
  if (op == m::operators::op_greater)            return OpKind::Gt;
  if (op == m::operators::op_greater_equals)     return OpKind::Ge;
  if (op == m::operators::op_equals_equals)      return OpKind::Eq;
  if (op == m::operators::op_exclamation_equals) return OpKind::Ne;
  if (op == m::operators::op_ampersand_ampersand) return OpKind::And;
  if (op == m::operators::op_pipe_pipe)           return OpKind::Or;
  throw "reflection AD: unsupported binary operator";
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
// The kinks. Real ops rather than desugared Selects, so `is_continuous_on`
// can see that |x| is continuous across zero. Max/Min follow std::max
// (`a < b ? b : a`), so a tie yields the first operand.
template <class T> inline T p_fabs(T x) { return std::fabs(x); }
template <class T> inline T p_abs (T x) { return std::abs(x); }
template <class T> inline T p_fmax(T x, T y) { return std::fmax(x, y); }
template <class T> inline T p_fmin(T x, T y) { return std::fmin(x, y); }
template <class T> inline T p_max (T x, T y) { return std::max(x, y); }
template <class T> inline T p_min (T x, T y) { return std::min(x, y); }

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
  if (is_std_fn(callee, ^^p_fabs<float>, ^^p_fabs<double>, ^^p_fabs<long double>) ||
      is_std_fn(callee, ^^p_abs<float>,  ^^p_abs<double>,  ^^p_abs<long double>))
    return {true, OpKind::Abs};
  if (is_std_fn(callee, ^^p_fmax<float>, ^^p_fmax<double>, ^^p_fmax<long double>) ||
      is_std_fn(callee, ^^p_max<float>,  ^^p_max<double>,  ^^p_max<long double>))
    return {true, OpKind::Max};
  if (is_std_fn(callee, ^^p_fmin<float>, ^^p_fmin<double>, ^^p_fmin<long double>) ||
      is_std_fn(callee, ^^p_min<float>,  ^^p_min<double>,  ^^p_min<long double>))
    return {true, OpKind::Min};

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
    // Reduce the literal to a value reflection (constant_of); it is spliced by
    // the existing P2996 value splice at codegen -- no expression splicing.
    return emit(c, OpKind::Const, 0, 0, m::constant_of(e));
  }

  if (m::is_unary_operator(e)) {
    m::operators op = m::expression_operator_of(e);
    std::size_t a = lower(c, m::operands_of(e).front());
    if (op == m::operators::op_plus)
      return a;  // unary plus is a no-op
    if (op == m::operators::op_minus)
      return emit(c, OpKind::Neg, a);
    if (op == m::operators::op_exclamation)
      return emit(c, OpKind::Not, a);
    throw "reflection AD: unsupported unary operator";
  }

  if (m::is_binary_operator(e)) {
    auto ops = m::operands_of(e);
    OpKind op = binOp(m::expression_operator_of(e));
    std::size_t a = lower(c, ops[0]);
    // Short-circuit: lower the right operand under the guard saying the left
    // did not already decide, so `x != 0 && 1/x > 5` stays safe.
    if (op == OpKind::And || op == OpKind::Or) {
      std::size_t outer = c.curGuard;
      std::size_t reached = (op == OpKind::And) ? a : emit(c, OpKind::Not, a);
      c.curGuard = narrow_guard(c, outer, reached);
      std::size_t b = lower(c, ops[1]);
      c.curGuard = outer;
      return emit(c, op, a, b);
    }
    std::size_t b = lower(c, ops[1]);
    return emit(c, op, a, b);
  }

  if (m::is_conditional_operator(e)) {
    // Each branch is lowered under a narrower guard, so its nodes exist in
    // the DAG but only evaluate when that branch is taken -- in all sweeps.
    std::size_t p = lower(c, m::condition_of(e));
    std::size_t outer = c.curGuard;

    c.curGuard = narrow_guard(c, outer, p);
    std::size_t whenTrue = lower(c, m::true_expression_of(e));

    c.curGuard = outer;
    std::size_t notP = emit(c, OpKind::Not, p);
    c.curGuard = narrow_guard(c, outer, notP);
    std::size_t whenFalse = lower(c, m::false_expression_of(e));

    c.curGuard = outer;
    return emit(c, OpKind::Select, whenTrue, whenFalse, ^^int, p);
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
      return emit(c, op, a, b);
    }
    std::size_t a = lower(c, ops[ops.size() - 1]);   // unary call
    return emit(c, op, a);
  }

  // Unsupported: emit a constant placeholder from its value.
  return emit(c, OpKind::Const, 0, 0, m::constant_of(e));
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
  return op == OpKind::Mul || op == OpKind::Div ||
         op == OpKind::Sin || op == OpKind::Cos || op == OpKind::Log ||
         op == OpKind::Erfc ||
         // pick an operand's tangent at runtime, so read the operands
         op == OpKind::Abs || op == OpKind::Max || op == OpKind::Min;
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
    else if (n.op == OpKind::Const || op_is_boolean(n.op))
      varied = false;   // a predicate is piecewise constant: zero derivative
    else
      // Select: v[a] || v[b]. The condition contributes no tangent.
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
    // The guard is read by whichever sweep touches this node, so it must be
    // computed even if nothing else wants its value.
    if (n.guard != UNGUARDED)
      need[n.guard] = 1;
    if (!n.vself)
      continue;
    if (deriv_reads_operand_vals(n.op)) {
      if (op_has_a(n.op)) need[n.a] = 1;
      if (op_has_b(n.op)) need[n.b] = 1;
    }
    if (deriv_reads_self_val(n.op))
      need[n.self] = 1;
    // A Select's derivative reads the condition even when its value is dead.
    if (op_has_cond(n.op))
      need[n.cond] = 1;
  }
  for (std::size_t i = N; i-- > 0;) {   // backward: needed val pulls in operands
    if (!need[i])
      continue;
    const Node &n = ns[i];
    if (op_has_a(n.op)) need[n.a] = 1;
    if (op_has_b(n.op)) need[n.b] = 1;
    if (op_has_cond(n.op)) need[n.cond] = 1;
    if (n.guard != UNGUARDED) need[n.guard] = 1;
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
// Higher-order derivatives by truncated Taylor propagation.
// ---------------------------------------------------------------------------
namespace detail {

// Raw append: always pushes a new node and returns its slot.
consteval std::size_t emit_raw(std::vector<Node> &out, OpKind op,
                               std::size_t a, std::size_t b, info leaf = ^^int,
                               std::size_t cond = 0,
                               std::size_t guard = UNGUARDED) {
  std::size_t s = out.size();
  out.push_back(Node{op, s, a, b, leaf, cond, guard});
  return s;
}

// If we have an existing node in the DAG with the same operation and operands, we
// can reuse it instead of bloating the DAG with duplicate nodes. Const nodes
// carry their value in `leaf` rather than in operands, so they only match when
// the leaf matches too. `cond` and `guard` are also keyed on since a value computed 
// under one guard must not be reused under another.
consteval std::size_t emit_node(std::vector<Node> &out, OpKind op,
                                std::size_t a, std::size_t b, info leaf = ^^int,
                                std::size_t cond = 0,
                                std::size_t guard = UNGUARDED) {
    for (const Node &n : out) {
      if (n.op == op && n.a == a && n.b == b && n.cond == cond && n.guard == guard &&
            (op != OpKind::Const || n.leaf == leaf))
        {
           return n.self;
        }
         
    }
    return emit_raw(out, op, a, b, leaf, cond, guard);
}

// This struct represents the shape of a truncated taylor polynomial, e.g.
// f(x, y) = f(a, b) + fx(a, b)(x-a) + fy(a, b)(y-b) + 1/2 * (fxx(a,b)(x-a)^2 + 2fxy(a,b)(x-a)(y-b) + fyy(a,b)(y-b)^2)
// Let's say f(x, y) is a combination of unary functions, and we want to calculate fxy(a, b) (1st derivative wrt x and y)
// We can look at the coefficient of (x-a)(y-b) in the taylor expansion of f(x, y) to work out fxy.
// e.g. f(x,y) = sin(x)cos(y) = sin(a)cos(b) + cos(a)cos(b)(x-a) - sin(a)sin(b)(y-b) + 0.5 (-sin(a)cos(b)(x-a)^2 - 2cos(a)sin(b)(x-a)^2 -sin(a)cos(b)(y-b)^2)
// we see the coefficient is -cos(a)sin(b)
// We can work this out in a simpler way, by considering f(x, y) = g(x)h(y)
// Where g(x) = sin(x) = sin(a) + cos(a)(x-a)
//       h(y) = cos(y) = cos(b) - sin(b)(y-b)
// We can multiply these together, and look at the coefficient of (x-a)(y-b) to get the same result.
// We use this trick to work out higher order complex derivatives, by composing simpler taylor polynomials.
struct PolynomialShape {
    // slot in the DAG of each argument we want to differentiate with respect to (e.g. for polynomial above, slot of x and y)
    std::vector<std::size_t> wrtSlots;              
    // how many terms in each polynomial are required for each slot. (e.g. above, we only needed 2 terms in the expansion of g(x) and h(y))
    // in general, we need the number of times we want to differentiate w.r.t + 1.
    std::vector<std::size_t> wrtTerms;              
    // The index of the term in the polynomial that corresponds to the first order derivative for each w.r.t.
    // e.g. for the example above firstDerivative = [1,2] 
    std::vector<std::size_t> firstDerivative;         
    // The order of each partial derivative for each term in the taylor polynomial
    // For instance in the example above, orders = {{0,0}, {1, 0}, {0, 1}, {1, 1}}. 
    std::vector<std::vector<std::size_t>> orders;   
    // how many terms are in the overall taylor polynomial (each wrt polynomical multiplied together)
    std::size_t totalTerms = 1;                     
    // the order of the overall taylor polynomial (e.g. x*x*y = 3rd order)
    std::size_t polynomialOrder = 0;    
    
    // For a polynomial with shape `sh`, is the algebraic term at index `a` a higher order than
    // that at index `b`. Here a higher oder means that `b` divides `a`, e.g. xy^2 is a divisor of
    // x^2y^3, but x^3y^2 is not.
    consteval bool isPolynomialDivisor(std::size_t a, std::size_t b) const {
        for (std::size_t i = 0; i < orders[a].size(); ++i) {
            if (orders[b][i] > orders[a][i]) {
                return false;
            }
        }
        return true;
    }

    // Create a `PolynomialShape` for a list of variables we want to differentiate `wrt`.
    static consteval PolynomialShape make_shape(const std::vector<std::size_t> &wrts) {
        PolynomialShape sh;
        for (std::size_t w : wrts) {
            bool seen = false;
            for (std::size_t i = 0; i < sh.wrtSlots.size(); ++i) {
                if (sh.wrtSlots[i] == w) { 
                    ++sh.wrtTerms[i];
                    seen = true;
                    break;
                }
            }
            if (!seen) { 
                sh.wrtSlots.push_back(w); sh.wrtTerms.push_back(2); 
            }
        }
        for (std::size_t r : sh.wrtTerms) { 
            sh.firstDerivative.push_back(sh.totalTerms); sh.totalTerms *= r; 
        }
        sh.polynomialOrder = wrts.size();
        sh.orders.resize(sh.totalTerms);
        for (std::size_t termIndex = 0; termIndex < sh.totalTerms; ++termIndex) {
            std::vector<std::size_t> degrees(sh.wrtSlots.size(), 0);
            std::size_t t = termIndex;
            // Arrange the polynomial degrees such that the term at index i can be created by multiplying 
            // terms at index j and (i-j) as long as the term at index j divides that at index i.
            // e.g. at index j we might have x^2 * y^2. At index i we have x^4 * y^5. Then at index i-j, we have
            // x^2 * y^3.
            // This is achieved by choosing the degrees via the `%` operator below.
            for (std::size_t i = 0; i < sh.wrtSlots.size(); ++i) {
                degrees[i] = t % sh.wrtTerms[i];
                t /= sh.wrtTerms[i];
            }
            sh.orders[termIndex] = degrees;
        }
        return sh;
    }
};

struct DAGBuilder {
  public:
    // Sentinel value to indicate that a derivative is zero and does not need adding to the DAG.
    static constexpr std::size_t kNoSlot = static_cast<std::size_t>(-1);

  private:
    // DAG slot of literal 1.0, if it's in the DAG.
    std::size_t one = kNoSlot;

    // A compile-time map from a constant scalar value to the slot of its (unique)
    // Const node.
    using ConstPool = std::vector<std::pair<double, std::size_t> >;
    ConstPool pool;

  public:
    std::vector<Node> out;

    // The guard every node emitted from here on is predicated on.
    std::size_t curGuard = UNGUARDED;

    // Return the slot of the Const node holding `value`. Cache it and ensure that there
    // are no duplicated constants in the output nodes. Constants do not need guarding, so
    // ignore the `curGuard`.
    consteval std::size_t ensure_const_node(double value) {
      for (const auto &[v, slot] : pool) {
          if (v == value) {
              return slot;
          }
      }
      std::size_t slot =
          emit_raw(out, OpKind::Const, 0, 0, m::reflect_constant(value));
      pool.push_back({value, slot});
      return slot;
    }

    // Add a node for `op` applied to the existing nodes at `slot1` (and, for a
    // binary op, `slot2`), under the guard currently in force.
    consteval std::size_t emit_op_node(OpKind op, std::size_t slot1, std::size_t slot2 = 0) {
        return emit_node(out, op, slot1, slot2, ^^int, 0, curGuard);
    }

    // Add a negation to the `out` vector, negating the existing node at position `slot`.
    consteval std::size_t emit_negation_node(std::size_t slot) {
        return slot == DAGBuilder::kNoSlot ? DAGBuilder::kNoSlot : emit_op_node(OpKind::Neg, slot, 0);
    }

    // Add an addition to the `out` vector, adding the existing nodes at positions `slot1` and `slot2`.
    consteval std::size_t emit_addition_node(std::size_t slot1, std::size_t slot2) {
        if (slot1 == DAGBuilder::kNoSlot) {
            return slot2;
        }
        if (slot2 == DAGBuilder::kNoSlot) {
            return slot1;
        }
        return emit_op_node(OpKind::Add, slot1, slot2);
    }

    // Add an subtraction to the `out` vector, subtracting the existing nodes at positions `slot1` and `slot2`.
    consteval std::size_t emit_subtraction_node(std::size_t slot1, std::size_t slot2) {
        if (slot2 == DAGBuilder::kNoSlot) {
            return slot1;
        }
        if (slot1 == DAGBuilder::kNoSlot) {
            return emit_negation_node(slot2);
        }
        return emit_op_node(OpKind::Sub, slot1, slot2);
    }

    // Add a multiplication to the `out` vector, multiplying the existing nodes at positions `slot1` and `slot2`.
    consteval std::size_t emit_multiplication_node(std::size_t slot1, std::size_t slot2) {
        if (slot1 == DAGBuilder::kNoSlot || slot2 == DAGBuilder::kNoSlot) {
            return DAGBuilder::kNoSlot;
        }
        if (slot1 == one) {
            return slot2;
        }
        if (slot2 == one) {
            return slot1;
        }
        return emit_op_node(OpKind::Mul, slot1, slot2);
    }

    // Add a division to the `out` vector, dividing the existing nodes at positions `slot1` and `slot2`.
    consteval std::size_t emit_division_node(std::size_t slot1, std::size_t slot2) {
        if (slot2 == DAGBuilder::kNoSlot) {
            throw "reflection AD: division by a structurally zero value";
        }
        if (slot1 == DAGBuilder::kNoSlot) {
            return DAGBuilder::kNoSlot;
        }
        if (slot2 == one) {
            return slot1;
        }
        return emit_op_node(OpKind::Div, slot1, slot2);
    }

    // Add a multiplcation to the `out` vector, scaling the node at position `slot` by the `scaleFactor`.
    consteval std::size_t emit_scaling_node(std::size_t slot, double scaleFactor) {
        if (slot == DAGBuilder::kNoSlot || scaleFactor == 0.0) {
            return DAGBuilder::kNoSlot;
        }
        if (scaleFactor == 1.0) {
            return slot;
        }
        if (scaleFactor == -1.0) {
            return emit_negation_node(slot);
        }
        return emit_op_node(OpKind::Mul, ensure_const_node(scaleFactor), slot);
    }

    // Add a `cond ? slot1 : slot2` to the `out` vector.
    consteval std::size_t emit_select_node(std::size_t cond, std::size_t slot1,
                                           std::size_t slot2) {
        if (slot1 == DAGBuilder::kNoSlot && slot2 == DAGBuilder::kNoSlot) {
            return DAGBuilder::kNoSlot;
        }
        std::size_t whenTrue  = slot1 == DAGBuilder::kNoSlot ? ensure_const_node(0.0) : slot1;
        std::size_t whenFalse = slot2 == DAGBuilder::kNoSlot ? ensure_const_node(0.0) : slot2;
        if (whenTrue == whenFalse) {
            // Both branches are the same, so no need to add a new node.
            return whenTrue;
        }
        return emit_node(out, OpKind::Select, whenTrue, whenFalse, ^^int, cond, curGuard);
    }

    // Return the slot in the DAG corresponding to the literal `1.0`.
    consteval std::size_t literal_one() {
        one = ensure_const_node(1.0);
        return one;
    }

    // For the DAG node at position `slot`, generate DAG nodes corresponding to powers of the
    // DAG node at `slot` up to and including the `order`th power. Return a vector containing
    // the position of these nodes in the DAG.
    consteval std::vector<std::size_t> pow_slots(std::size_t slot, std::size_t order) {
        std::vector<std::size_t> powers;
        powers.push_back(literal_one());
        for (std::size_t m = 1; m <= order; ++m) {
            powers.push_back(emit_multiplication_node(powers[m - 1], slot));
        }
        return powers;
    }
};

// One slot per multi-index, in rank order; DAGBuilder::kNoSlot where the coefficient is
// statically zero.
using Polynomial = std::vector<std::size_t>;

struct PolynomialBuilder {
    // Create a polynomial of the specified shape, with every coefficient zero.
    static consteval Polynomial createZeroPolynomial(const PolynomialShape &shape) { return Polynomial(shape.totalTerms, DAGBuilder::kNoSlot); }

    // Multiply polynomials `A` and `B`, both of the shape `sh` and return the result. All coefficients of the result
    // are added to the `builder`.
    static consteval Polynomial multiplyPolynomials(DAGBuilder &builder, const PolynomialShape &sh, const Polynomial &A, const Polynomial &B) {
        Polynomial result = PolynomialBuilder::createZeroPolynomial(sh);
        // Calculate each term in the result polynomial one by one.
        for (std::size_t currentTerm = 0; currentTerm < sh.totalTerms; ++currentTerm) {
            // calculate the current coefficient by looking through all terms in both polynomials that are factors.
            // e.g. (1 + 3x + 4x^2) * (2 + 2x + 5x^2)
            // the coefficient of x^2 in the result comes from (1*5 + 3*2 + 4 * 2)
            std::size_t coefficient = DAGBuilder::kNoSlot;
            for (std::size_t factorTerm = 0; factorTerm <= currentTerm; ++factorTerm) {          
                if (A[factorTerm] == DAGBuilder::kNoSlot || B[currentTerm - factorTerm] == DAGBuilder::kNoSlot || !sh.isPolynomialDivisor(currentTerm, factorTerm)) {
                    continue;
                }
                // by construction, (polynomial power at index b) * (polynomial power at index a - b) = (polynomial power at index a)
                coefficient = builder.emit_addition_node(coefficient, builder.emit_multiplication_node(A[factorTerm], B[currentTerm - factorTerm]));
            }
            result[currentTerm] = coefficient;
        }
        return result;
    }

    // Divide polynomial `A` by `B`, both of the shape `sh` and return the result. All coefficients of the result
    // are added to the `builder`.
    static consteval Polynomial dividePolynomials(DAGBuilder &builder, const PolynomialShape &sh, const Polynomial &A, const Polynomial &B) {
        if (B[0] == DAGBuilder::kNoSlot) {
            throw "reflection AD: division by a structurally zero value";
        }
        Polynomial result = PolynomialBuilder::createZeroPolynomial(sh);
        // Start with the 0th order term of `A`. Only the 0th order term of `B` can divide it. This fixes the 0th order
        // term in the result.
        // We can then consider the 1st order term of `A`. Only the 0th and 1st order terms of `B` can divide it, where
        // the first order term of `B` is multiplied by the 0th order term of `result`, and the 0th order term of `B` is
        // multiplied by the 1st order term of result. We use this to work out the coefficient of the first order term of
        // result. We can use a similar pattern to work out all coefficients.
        for (std::size_t termIndex = 0; termIndex < sh.totalTerms; ++termIndex) {
            std::size_t coefficient = A[termIndex];
            for (std::size_t factorTerm = 1; factorTerm <= termIndex; ++factorTerm) {
                if (B[factorTerm] == DAGBuilder::kNoSlot || result[termIndex - factorTerm] == DAGBuilder::kNoSlot || !sh.isPolynomialDivisor(termIndex, factorTerm)) {
                    continue;
                }
                coefficient = builder.emit_subtraction_node(coefficient, builder.emit_multiplication_node(B[factorTerm], result[termIndex - factorTerm]));
            }
            result[termIndex] = builder.emit_division_node(coefficient, B[0]);
        }
        return result;
    }

};

// Return the coefficients of the taylor expansion for the unary function `op` around
// `a0`, up to the `order` specified.
consteval std::vector<std::size_t> calculateTaylorCoefficients(DAGBuilder &builder, OpKind op,
                                                std::size_t a0,
                                                std::size_t order) {
    // Coefficients of the taylor polynomial.
    std::vector<std::size_t> coefficients(order + 1, DAGBuilder::kNoSlot);

    // Create a vector, where index i = 1/i!
    std::vector<double> inv_factorial(order + 1, 1.0);
    for (std::size_t m = 1; m <= order; ++m) {
        inv_factorial[m] = inv_factorial[m - 1] / static_cast<double>(m);
    }
      
    switch (op) {
        case OpKind::Exp: {
            // exp(x) = exp(a0) + exp(a0)/1! * (x-a0) + exp(a0)/2! * (x-a0)^2 ...
            std::size_t e = builder.emit_op_node(OpKind::Exp, a0);
            for (std::size_t m = 0; m <= order; ++m)
              coefficients[m] = builder.emit_scaling_node(e, inv_factorial[m]);
            break;
        }
        case OpKind::Sin:
        case OpKind::Cos: {
            // sin(x) = sin(a0) + cos(a0)/1! * (x-a0) - sin(a0)/2! * (x-a0)^2 ...
            // cos(x) = cos(a0) - sin(a0)/1! * (x-a0) - cos(a0)/2! * (x-a0)^2 ...
            std::size_t s = builder.emit_op_node(OpKind::Sin, a0);
            std::size_t k = builder.emit_op_node(OpKind::Cos, a0);
            for (std::size_t m = 0; m <= order; ++m) {
              std::size_t phase = ((op == OpKind::Sin ? m : m + 1) % 4);
              std::size_t base = (phase % 2 == 0) ? s : k;
              double sign = (phase < 2) ? 1.0 : -1.0;
              coefficients[m] = builder.emit_scaling_node(base, sign * inv_factorial[m]);
          }
          break;
        }
        case OpKind::Log: {
            // log(x) = log(a0) + 1/(1! * a0) * (x-a0) - 1/(2! * a0^2) * (x-a0)^2 ... + (-1)^(n+1)/(n * a0^n) * (x-a0)^n
            coefficients[0] = builder.emit_op_node(OpKind::Log, a0);
            std::vector<std::size_t> p = builder.pow_slots(a0, order);
            for (std::size_t m = 1; m <= order; ++m) {
                double k = ((m % 2) ? 1.0 : -1.0) / static_cast<double>(m);
                coefficients[m] = builder.emit_division_node(builder.ensure_const_node(k), p[m]);
            }
            break;
        }
      case OpKind::Sqrt: { 
          // sqrt(x) = sqrt(a0) + 1/1!*sqrt(a0)/2*a0 (x-a0) + 1/2!*sqrt(a0)/4*a0^2 (x-a0)^2 +... + 1/n!(-1)^n-1(2n-3)!/2^n * sqrt(a0)/a0^n * (x-a0)^n
          std::size_t s = builder.emit_op_node(OpKind::Sqrt, a0);
          coefficients[0] = s;
          // Need to divide by a0^n for the nth term, so calculate powers.
          std::vector<std::size_t> powers = builder.pow_slots(a0, order);
          double coefficient = -1.0;
          for (std::size_t m = 1; m <= order; ++m) {
              coefficient *= -0.5 * static_cast<double>(3 - 2*m) / static_cast<double>(m);
              coefficients[m] = builder.emit_scaling_node(builder.emit_division_node(s, powers[m]), coefficient);
          }
          break;
      }
      case OpKind::Erfc: {
          // mth derivative of erfc:
          // erfc^(m)(a) = -2/sqrt(pi) * (-1)^(m-1) * H_{m-1}(a) * exp(-a*a)
          // where H_{m-1} is the m-1th physicist's Hermite polynomial
          // and H_{n+1}(x) = 2x * H_{n}(x) - 2*n*H_{n-1}(x)
          // and H_0(x) = 1, H_1(x) = 2x
          coefficients[0] = builder.emit_op_node(OpKind::Erfc, a0);
          if (order >= 1) {
              std::size_t square = builder.emit_op_node(OpKind::Mul, a0, a0);
              std::size_t e = builder.emit_op_node(OpKind::Exp,
                                                  builder.emit_op_node(OpKind::Neg, square));
              std::vector<std::size_t> hermitePolynomials(order, DAGBuilder::kNoSlot);
              hermitePolynomials[0] = builder.literal_one();                                    // H_0 = 1
              if (order >= 2) hermitePolynomials[1] = builder.emit_scaling_node(a0, 2.0);           // H_1 = 2a
              for (std::size_t n = 2; n < order; ++n) {
                  // Calculate each Hermite polynomial.
                  std::size_t two_times_a = builder.emit_scaling_node(a0, 2.0);
                  std::size_t part1 = builder.emit_multiplication_node(two_times_a, hermitePolynomials[n - 1]);
                  std::size_t part2 = builder.emit_scaling_node(hermitePolynomials[n - 2], 2.0 * static_cast<double>(n - 1));
                  hermitePolynomials[n] = builder.emit_subtraction_node(part1, part2);
              }
                
              for (std::size_t m = 1; m <= order; ++m) {
                  double k = -two_over_root_pi * ((m % 2) ? 1.0 : -1.0) * inv_factorial[m];
                  coefficients[m] = builder.emit_scaling_node(builder.emit_multiplication_node(hermitePolynomials[m - 1], e), k);
              }
          }
          break;
      }
      default:
         throw "reflection AD: unsupported operation";
    }
    return coefficients;
}

// Use the chain rule to calculate the coefficients of a Taylor polynomial for the Operation `op`
// around A[0], where `A` is the taylor polynomial of the input of `op`. We can substitute in the
// taylor polynomial for `A` into the taylor expansion for `op`.
// E.g. exp(sin(x))
//  exp(x) = exp(a0) + exp(a0)/1! * (x-a0) + exp(a0)/2! * (x-a0)^2 ...   (1)
//  sin(y) = sin(b0) + cos(b0)/1! * (y-b0) - sin(b0)/2! * (y-b0)^2 ...   (2)
// Putting x = sin(y), a0 = sin(b0) into (1)
//  exp(sin(y)) = exp(sin(b0)) + exp(sin(b0))/1! * (cos(b0)/1! * (y-b0) - sin(b0)/2! * (y-b0)^2) + exp(sin(b0))/2! + (cos(b0)/1! * (y-b0) - sin(b0)/2! * (y-b0)^2) ^ 2 ...
//              = exp(sin(b0)) + exp(sin(b0))/1! * (cos(b0)/1! * (y-b0)) to first order.
consteval Polynomial apply_unary_chain_rule(DAGBuilder &builder, const PolynomialShape &sh, OpKind op, const Polynomial &A) {
    std::vector<std::size_t> coefficients = calculateTaylorCoefficients(builder, op, A[0], sh.polynomialOrder);
    Polynomial result = PolynomialBuilder::createZeroPolynomial(sh);
    result[0] = coefficients[0];

    // Let P be the taylor polynomial of the input to `op`, `A`, expanded around A[0]
    Polynomial P = A;
    // Set P[0] to 0, meaning P = A - A[0].
    P[0] = DAGBuilder::kNoSlot;

    // We can calculate the taylor polynomial R: sum( coefficients[i] * (A - A[0]) ^ i-1 )
    // We can substitute in P to get the taylor polynomial for R in terms of the inputs to A.
    Polynomial pw = PolynomialBuilder::createZeroPolynomial(sh);
    pw[0] = builder.literal_one();                       // P^0
    // For each polynomial order we care about, multiple `pw` by `P` and update the coefficients.
    // We need
    //  to do this for each power, since for instance if P = x + x^2, and we care about 
    // 4th order terms, we'll get contributions from P^2 and P^4.
    for (std::size_t m = 1; m <= sh.polynomialOrder; ++m) {
        pw = PolynomialBuilder::multiplyPolynomials(builder, sh, pw, P);
        bool any = false;
        for (std::size_t r = 0; r < sh.totalTerms; ++r) {
            if (pw[r] != DAGBuilder::kNoSlot) { any = true; break; }
        }
        if (!any) {
            break;
        }
        for (std::size_t r = 0; r < sh.totalTerms; ++r) {
            result[r] = builder.emit_addition_node( result[r], builder.emit_multiplication_node(coefficients[m], pw[r]));
        }
    }
    return result;
}

// Build the DAG computing d^|Wrts| f / prod(d x_w). For each node in `src`, compute the taylor
// expansion of the required order. Combine these taylor polynomials according to the DAG structure
// and extract the derivatives from the result.
consteval std::vector<Node> build_taylor_nodes(const std::vector<Node> &src,
                                            const std::vector<std::size_t> &wrts) {
    PolynomialShape sh = PolynomialShape::make_shape(wrts);
    DAGBuilder builder;

    // Inputs come first and keep their slots: the evaluator reads `in[n.self]`.
    // Emitting them before any constant is what preserves that.
    for (const Node &n : src) {
        if (n.op == OpKind::Input) {
            std::size_t s = emit_raw(builder.out, OpKind::Input, 0, 0);
            if (s != n.self)
               throw "reflection AD: input nodes are not at the head of the DAG";
        }
    }
  
    // Create a taylor polynomial for each node in the src.
    std::vector<Polynomial> polynomials(src.size());
    for (const Node &n : src) {
      Polynomial current = PolynomialBuilder::createZeroPolynomial(sh);
      // The coefficient arithmetic for this node belongs to the same branch the
      // node itself does. `src` guards are slots in `src`; the corresponding
      // value in the DAG being built is the guard's 0th coefficient, and guards
      // precede what they guard, so it is already there.
      builder.curGuard = n.guard == UNGUARDED ? UNGUARDED : polynomials[n.guard][0];
      switch (n.op) {
        case OpKind::Input:
          // Construct the taylor polynomial for the input node. The first term of the polynomial
          // is the value itself. The first derivative is 1, if we differentiate this argument
          // and every other term is 0.
          current[0] = n.self;
          for (std::size_t i = 0; i < sh.wrtSlots.size(); ++i) {
              if (sh.wrtSlots[i] == n.self) {
                 current[sh.firstDerivative[i]] = builder.literal_one();
              }            
          }
          break;
        case OpKind::Const:
          // We don't use the DAG builder const pool here, as we wish to preserve the constant from the
          // original calculation.
          current[0] = emit_raw(builder.out, OpKind::Const, 0, 0, n.leaf);
          break;
        case OpKind::Output:
          // Add a new polynomial to indicate the output.
          current = polynomials[n.a];
          break;
        case OpKind::Add:
          // Add the coefficients of the two polynomials together.
          for (std::size_t r = 0; r < sh.totalTerms; ++r) {
              current[r] = builder.emit_addition_node(polynomials[n.a][r], polynomials[n.b][r]);
          }
          break;
        case OpKind::Sub:
          // Subtract the coefficients of the second polynomial from the first.
          for (std::size_t r = 0; r < sh.totalTerms; ++r) {
              current[r] = builder.emit_subtraction_node(polynomials[n.a][r], polynomials[n.b][r]);
          }
          break;
        case OpKind::Neg:
          // Negate every coefficient of the polynomial.
          for (std::size_t r = 0; r < sh.totalTerms; ++r) {
              current[r] = builder.emit_negation_node(polynomials[n.a][r]);
          }
          break;
        case OpKind::Mul:
          current = PolynomialBuilder::multiplyPolynomials(builder, sh, polynomials[n.a], polynomials[n.b]);
          break;
        case OpKind::Div:
          current = PolynomialBuilder::dividePolynomials(builder, sh, polynomials[n.a], polynomials[n.b]);
          break;
        case OpKind::Lt: case OpKind::Le: case OpKind::Gt: case OpKind::Ge:
        case OpKind::Eq: case OpKind::Ne:
        case OpKind::And: case OpKind::Or: case OpKind::Not:
          // Propagate the condition, acting on the 0th order term. Higher orders are zero.
          current[0] = builder.emit_op_node(n.op, polynomials[n.a][0],
                                      op_has_b(n.op) ? polynomials[n.b][0] : 0);
          break;
        case OpKind::Select:
          // The current polynomial is set to the selected polynomial.
          for (std::size_t r = 0; r < sh.totalTerms; ++r)
            current[r] = builder.emit_select_node(polynomials[n.cond][0], polynomials[n.a][r], polynomials[n.b][r]);
          break;
        case OpKind::Abs: {
          // Take the absolute value of the input polynomial.
          const Polynomial &A = polynomials[n.a];
          std::size_t isNeg =
              builder.emit_op_node(OpKind::Lt, A[0], builder.ensure_const_node(0.0));
          current[0] = builder.emit_op_node(OpKind::Abs, A[0]);
          for (std::size_t r = 1; r < sh.totalTerms; ++r)
            current[r] = builder.emit_select_node(isNeg, builder.emit_negation_node(A[r]), A[r]);
          break;
        }
        case OpKind::Max:  
        case OpKind::Min: {     
          // Take whichever polynomial has a higher 0th order term.
          const Polynomial &A = polynomials[n.a];
          const Polynomial &B = polynomials[n.b];
          std::size_t cmp = (n.op == OpKind::Max)
              ? builder.emit_op_node(OpKind::Lt, A[0], B[0])
              : builder.emit_op_node(OpKind::Lt, B[0], A[0]);
          current[0] = builder.emit_op_node(n.op, A[0], B[0]);
          for (std::size_t r = 1; r < sh.totalTerms; ++r)
            current[r] = builder.emit_select_node(cmp, B[r], A[r]);
          break;
        }
        default:                                   // unary primitives (throws if not)
          current = apply_unary_chain_rule(builder, sh, n.op, polynomials[n.a]);
          break;
      }
      polynomials[n.self] = current;
    }

    // The root is outside every branch, so the scaling below is unconditional.
    builder.curGuard = UNGUARDED;

    // Output polynomial is the last one.
    const Polynomial& outputPolynomial = polynomials[src.size() - 1];

    // Requested derivative is the highest power in the polynomial by construction.
    std::size_t top = outputPolynomial[sh.totalTerms - 1];
    // Coefficient of the highest power in the polynomial is of the form 1/k!(d^k/dx) * 1/j!(d^j/dy)...
    // So need to multiply out the factorials to retrieve the derivative.
    double factorials = 1.0;
    for (std::size_t r : sh.wrtTerms) {
        for (std::size_t k = 2; k < r; ++k) {
          factorials *= static_cast<double>(k);
        }
    }

    // Add a node to the DAG to represent the output derivative.
    std::size_t res = builder.emit_scaling_node(top, factorials);
    if (res == DAGBuilder::kNoSlot)                            // derivative is statically zero
      res = builder.ensure_const_node(0.0);
    emit_raw(builder.out, OpKind::Output, res, 0);
    return builder.out;
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
    if (op_has_cond(n.op)) need[n.cond] = 1;
    if (n.guard != UNGUARDED) need[n.guard] = 1;
  }
  for (Node &n : ns)
    n.nself = need[n.self];
}

// The value rule for one node, shared by all three sweeps so adding an op is a
// one-place change. `nself`: a value no emitted derivative reads is skipped.
template <Node N, typename T>
constexpr void eval_primal(T *val, const T *in) {
  if constexpr (N.nself) {
    // A guarded node belongs to a branch; skip it unless that branch is taken.
    if constexpr (N.guard != UNGUARDED) {
      if (!(val[N.guard] != T{0}))
        return;
    }
    if constexpr (N.op == OpKind::Input)       val[N.self] = in[N.self];
    else if constexpr (N.op == OpKind::Const)  val[N.self] = static_cast<T>([: N.leaf :]);
    else if constexpr (N.op == OpKind::Output) val[N.self] = val[N.a];
    else if constexpr (N.op == OpKind::Add)    val[N.self] = val[N.a] + val[N.b];
    else if constexpr (N.op == OpKind::Sub)    val[N.self] = val[N.a] - val[N.b];
    else if constexpr (N.op == OpKind::Mul)    val[N.self] = val[N.a] * val[N.b];
    else if constexpr (N.op == OpKind::Div)    val[N.self] = val[N.a] / val[N.b];
    else if constexpr (N.op == OpKind::Neg)    val[N.self] = -val[N.a];
    else if constexpr (N.op == OpKind::Sin)    val[N.self] = std::sin(val[N.a]);
    else if constexpr (N.op == OpKind::Cos)    val[N.self] = std::cos(val[N.a]);
    else if constexpr (N.op == OpKind::Exp)    val[N.self] = std::exp(val[N.a]);
    else if constexpr (N.op == OpKind::Log)    val[N.self] = std::log(val[N.a]);
    else if constexpr (N.op == OpKind::Sqrt)   val[N.self] = std::sqrt(val[N.a]);
    else if constexpr (N.op == OpKind::Erfc)   val[N.self] = std::erfc(val[N.a]);
    else if constexpr (N.op == OpKind::Lt)     val[N.self] = (val[N.a] <  val[N.b]) ? T{1} : T{0};
    else if constexpr (N.op == OpKind::Le)     val[N.self] = (val[N.a] <= val[N.b]) ? T{1} : T{0};
    else if constexpr (N.op == OpKind::Gt)     val[N.self] = (val[N.a] >  val[N.b]) ? T{1} : T{0};
    else if constexpr (N.op == OpKind::Ge)     val[N.self] = (val[N.a] >= val[N.b]) ? T{1} : T{0};
    else if constexpr (N.op == OpKind::Eq)     val[N.self] = (val[N.a] == val[N.b]) ? T{1} : T{0};
    else if constexpr (N.op == OpKind::Ne)     val[N.self] = (val[N.a] != val[N.b]) ? T{1} : T{0};
    else if constexpr (N.op == OpKind::Not)    val[N.self] = (val[N.a] != T{0}) ? T{0} : T{1};
    // Reads val[b] only when val[a] leaves it undecided -- exactly when the
    // right operand was lowered as reachable, so its slot is written.
    else if constexpr (N.op == OpKind::And)
      val[N.self] = (val[N.a] != T{0} && val[N.b] != T{0}) ? T{1} : T{0};
    else if constexpr (N.op == OpKind::Or)
      val[N.self] = (val[N.a] != T{0} || val[N.b] != T{0}) ? T{1} : T{0};
    // Select likewise reads only the branch it takes.
    else if constexpr (N.op == OpKind::Select)
      val[N.self] = (val[N.cond] != T{0}) ? val[N.a] : val[N.b];
    else if constexpr (N.op == OpKind::Abs)
      val[N.self] = (val[N.a] < T{0}) ? -val[N.a] : val[N.a];
    else if constexpr (N.op == OpKind::Max)
      val[N.self] = (val[N.a] < val[N.b]) ? val[N.b] : val[N.a];
    else if constexpr (N.op == OpKind::Min)
      val[N.self] = (val[N.b] < val[N.a]) ? val[N.b] : val[N.a];
  }
}

// Forward-mode tangent for one node, guarded like the primal so an untaken
// branch contributes no tangent work.
template <Node N, typename T, std::size_t Wrt>
constexpr void eval_tangent(T *tang, const T *val) {
  if constexpr (N.op == OpKind::Input) {
    tang[N.self] = (N.self == Wrt) ? T{1} : T{0};
  } else if constexpr (!N.vself) {
    tang[N.self] = T{0};   // not varied (includes Const and every predicate)
  } else {
    if constexpr (N.guard != UNGUARDED) {
      if (!(val[N.guard] != T{0}))
        return;
    }
    if constexpr (N.op == OpKind::Output)    tang[N.self] = tang[N.a];
    else if constexpr (N.op == OpKind::Neg)  tang[N.self] = -tang[N.a];
    else if constexpr (N.op == OpKind::Sin)  tang[N.self] = std::cos(val[N.a]) * tang[N.a];
    else if constexpr (N.op == OpKind::Cos)  tang[N.self] = -std::sin(val[N.a]) * tang[N.a];
    else if constexpr (N.op == OpKind::Exp)  tang[N.self] = val[N.self] * tang[N.a];
    else if constexpr (N.op == OpKind::Log)  tang[N.self] = tang[N.a] / val[N.a];
    else if constexpr (N.op == OpKind::Sqrt) tang[N.self] = tang[N.a] / (T{2} * val[N.self]);
    else if constexpr (N.op == OpKind::Erfc)
      tang[N.self] = tang[N.a] * -1 * two_over_root_pi * (std::exp(-1 * (val[N.a] * val[N.a])));
    else if constexpr (N.op == OpKind::Add) {
      if constexpr (N.va && N.vb) tang[N.self] = tang[N.a] + tang[N.b];
      else if constexpr (N.va)    tang[N.self] = tang[N.a];
      else                        tang[N.self] = tang[N.b];
    } else if constexpr (N.op == OpKind::Sub) {
      if constexpr (N.va && N.vb) tang[N.self] = tang[N.a] - tang[N.b];
      else if constexpr (N.va)    tang[N.self] = tang[N.a];
      else                        tang[N.self] = -tang[N.b];
    } else if constexpr (N.op == OpKind::Mul) {
      if constexpr (N.va && N.vb) tang[N.self] = tang[N.a] * val[N.b] + val[N.a] * tang[N.b];
      else if constexpr (N.va)    tang[N.self] = tang[N.a] * val[N.b];
      else                        tang[N.self] = val[N.a] * tang[N.b];
    } else if constexpr (N.op == OpKind::Div) {
      if constexpr (N.va && N.vb)
        tang[N.self] = (tang[N.a] * val[N.b] - val[N.a] * tang[N.b]) / (val[N.b] * val[N.b]);
      else if constexpr (N.va)    tang[N.self] = tang[N.a] / val[N.b];
      else                        tang[N.self] = -val[N.a] * tang[N.b] / (val[N.b] * val[N.b]);
    } else if constexpr (N.op == OpKind::Select) {
      if constexpr (N.va && N.vb) tang[N.self] = (val[N.cond] != T{0}) ? tang[N.a] : tang[N.b];
      else if constexpr (N.va)    tang[N.self] = (val[N.cond] != T{0}) ? tang[N.a] : T{0};
      else                        tang[N.self] = (val[N.cond] != T{0}) ? T{0} : tang[N.b];
    } else if constexpr (N.op == OpKind::Abs) {
      tang[N.self] = (val[N.a] < T{0}) ? -tang[N.a] : tang[N.a];
    } else if constexpr (N.op == OpKind::Max || N.op == OpKind::Min) {
      // Both pass one operand through; they differ only in which way round.
      const bool takes_b = (N.op == OpKind::Max) ? (val[N.a] < val[N.b])
                                                : (val[N.b] < val[N.a]);
      if constexpr (N.va && N.vb) tang[N.self] = takes_b ? tang[N.b] : tang[N.a];
      else if constexpr (N.va)    tang[N.self] = takes_b ? T{0} : tang[N.a];
      else                        tang[N.self] = takes_b ? tang[N.b] : T{0};
    } else {
      tang[N.self] = T{0};   // any unhandled op
    }
  }
}

// Reverse mode: push this node's adjoint to its operands via the local VJP.
// Varied operands only (others are a wasted `+= ... * 0`), guard permitting.
template <Node N, typename T>
constexpr void eval_adjoint(T *adj, const T *val) {
  // Nothing to push: bail before the guard load, and before Max/Min reads
  // operand values mark_activity never marked as needed.
  if constexpr (!N.va && !N.vb)
    return;
  else {
  if constexpr (N.guard != UNGUARDED) {
    if (!(val[N.guard] != T{0}))
      return;
  }
  if constexpr (N.op == OpKind::Output) {
    if constexpr (N.va) adj[N.a] += adj[N.self];
  } else if constexpr (N.op == OpKind::Add) {
    if constexpr (N.va) adj[N.a] += adj[N.self];
    if constexpr (N.vb) adj[N.b] += adj[N.self];
  } else if constexpr (N.op == OpKind::Sub) {
    if constexpr (N.va) adj[N.a] += adj[N.self];
    if constexpr (N.vb) adj[N.b] -= adj[N.self];
  } else if constexpr (N.op == OpKind::Mul) {
    if constexpr (N.va) adj[N.a] += adj[N.self] * val[N.b];
    if constexpr (N.vb) adj[N.b] += adj[N.self] * val[N.a];
  } else if constexpr (N.op == OpKind::Div) {
    if constexpr (N.va) adj[N.a] += adj[N.self] / val[N.b];
    if constexpr (N.vb) adj[N.b] -= adj[N.self] * val[N.a] / (val[N.b] * val[N.b]);
  } else if constexpr (N.op == OpKind::Neg) {
    if constexpr (N.va) adj[N.a] -= adj[N.self];
  } else if constexpr (N.op == OpKind::Sin) {
    if constexpr (N.va) adj[N.a] += adj[N.self] * std::cos(val[N.a]);
  } else if constexpr (N.op == OpKind::Cos) {
    if constexpr (N.va) adj[N.a] += -adj[N.self] * std::sin(val[N.a]);
  } else if constexpr (N.op == OpKind::Exp) {
    if constexpr (N.va) adj[N.a] += adj[N.self] * val[N.self];
  } else if constexpr (N.op == OpKind::Log) {
    if constexpr (N.va) adj[N.a] += adj[N.self] / val[N.a];
  } else if constexpr (N.op == OpKind::Sqrt) {
    if constexpr (N.va) adj[N.a] += adj[N.self] / (T{2} * val[N.self]);
  } else if constexpr (N.op == OpKind::Erfc) {
    if constexpr (N.va)
      adj[N.a] += adj[N.self] * -1 * two_over_root_pi * (std::exp(-1 * (val[N.a] * val[N.a])));
  } else if constexpr (N.op == OpKind::Select) {
    // The adjoint flows only down the branch that was taken.
    if constexpr (N.va) { if (val[N.cond] != T{0})    adj[N.a] += adj[N.self]; }
    if constexpr (N.vb) { if (!(val[N.cond] != T{0})) adj[N.b] += adj[N.self]; }
  } else if constexpr (N.op == OpKind::Abs) {
    if constexpr (N.va) adj[N.a] += (val[N.a] < T{0}) ? -adj[N.self] : adj[N.self];
  } else if constexpr (N.op == OpKind::Max || N.op == OpKind::Min) {
    const bool takes_b = (N.op == OpKind::Max) ? (val[N.a] < val[N.b])
                                              : (val[N.b] < val[N.a]);
    if constexpr (N.va) { if (!takes_b) adj[N.a] += adj[N.self]; }
    if constexpr (N.vb) { if (takes_b)  adj[N.b] += adj[N.self]; }
  }
  }
}

}  // namespace detail

// Create nodes for a DAG corresponding to the partial derivative of `Fn`,
// differentiated with respect to each argument in `Wrts...`.
template <info Fn, std::size_t... Wrts>
consteval std::vector<Node> build_partial_nodes() {
  std::vector<Node> ns = detail::build_taylor_nodes(
      build_nodes<Fn>(), std::vector<std::size_t>{ Wrts... });
  detail::prune_reachable(ns);
  return ns;
}

// Partial derivative of `Fn` with respect to each index in `Wrts...`, evaluated at `Arg...`.
template <info Fn, std::size_t... Wrts, typename... Args>
constexpr double partial_derivative(Args... args) {
  static constexpr auto nodes = std::define_static_array(build_partial_nodes<Fn, Wrts...>());
  constexpr std::size_t N = nodes.size();
  const double in[] = { static_cast<double>(args)... };
  double val[N] = {};
  template for (constexpr auto n : nodes) {
    detail::eval_primal<n, double>(val, in);
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
  T val[N] = {};
  T tang[N] = {};

  template for (constexpr auto n : nodes) {
    // Primal (only where the value is actually read by a derivative).
    detail::eval_primal<n, T>(val, in);

    // Tangent (activity-gated: a non-varied operand contributes a zero term,
    // which is dropped instead of emitted as `... * 0`).
    detail::eval_tangent<n, T, Wrt>(tang, val);
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
  T val[N] = {};
  T adj[N] = {};   // adjoints, accumulated; zero-initialized

  // Forward (primal) sweep: compute the values the adjoint rules will read.
  template for (constexpr auto n : nodes) {
    detail::eval_primal<n, T>(val, in);
  }

  // Seed the output adjoint, then sweep the DAG in reverse, pushing each node's
  // adjoint to its operands via the local VJP (accumulating with +=).
  adj[N - 1] = T{1};
  template for (constexpr auto n : rnodes) {
    detail::eval_adjoint<n, T>(adj, val);
  }

  // Input k's accumulated adjoint is the k-th partial derivative.
  std::array<T, P> g{};
  for (std::size_t i = 0; i < P; ++i)
    g[i] = adj[i];
  return g;
}

}  // namespace ad

#endif  // REFLECT_DEMO_AUTOGRAD_H
