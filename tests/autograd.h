// autograd.h — compile-time automatic differentiation via reflection.
//
// Pick a straight-line scalar function, reflect its body, and get a zero-cost
// spliced derivative. Built on the statement/expression reflection extension to
// P2996 (body_of / statements_of / return_value_of / ... plus the existing
// expression reflection).
//
// Design (see docs/ + plan): reflection -> SSA/let-DAG IR -> array-tape
// codegen. Each statement of the body becomes SSA nodes; sweeps over runtime
// arrays are unrolled with an expansion statement, indexing the in-scope arrays
// directly (only literal leaves are spliced).
//
// Modes:
//   - forward_derivative<^^f, Wrt>(args...)  — forward mode (JVP), one partial.
//   - gradient_of<^^f>(args...)              — full gradient via P forward
//   passes.
//   - gradient_reverse<^^f>(args...)         — full gradient in ONE reverse
//   pass
//       (primal sweep + adjoint sweep over the reversed DAG with accumulation);
//       the efficient path for scalar-output, many-input functions.
//   - partial_derivative<^^f, Wrt...>(args...) - Calculate the
//   partial_derivative of `f`, with respect to each index in `Wrt...`
//        e.g. partial_derivative<^^f, 0, 0, 1>(args...) will differentiate `f`
//        twice w.r.t the 0th argument, and once w.r.t. the first.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <meta>
#include <numbers>
#include <string_view>
#include <utility>
#include <vector>

namespace ad {
using std::meta::info;
namespace m = std::meta;

using namespace std::numbers;
constexpr double two_over_root_pi = 2. * std::numbers::inv_sqrtpi_v<double>;

// ---------------------------------------------------------------------------
// IR
// ---------------------------------------------------------------------------
enum class OpKind {
  Input,
  Const,
  Output,
  Add,
  Sub,
  Mul,
  Div,
  Neg,
  Sin,
  Cos,
  Exp,
  Log,
  Sqrt,
  Erfc,
  // Conditions and their combinators. Piecewise constant, so never "varied";
  // valued 0 or 1 in the same array as everything else.
  Lt,
  Le,
  Gt,
  Ge,
  Eq,
  Ne,
  And,
  Or,
  Not,
  // Select is `c ? a : b`; Abs/Max/Min are the kinks.
  Select,
  Abs,
  Max,
  Min,
  // Tensor ops (recognised as named calls; VJPs live in the tensor engine).
  Matmul,
  Transpose,
  Sum,
  Relu,
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
template <info F> struct primitive; // declared, never defined

// `^^f` is ill-formed when `f` names an overload set, so a specific overload is
// selected by signature:
//   template <> struct ad::primitive<ad::overload_of(^^nn, "sum",
//   ^^Tensor(const Tensor &))>
consteval info overload_of(info scope, std::string_view name, info fnType) {
  for (info mem : m::members_of(scope, m::access_context::current()))
    if (m::has_identifier(mem) && m::identifier_of(mem) == name &&
        m::type_of(mem) == fnType)
      return mem;
  throw "reflection AD: no overload with that signature in this scope";
}

struct Node {
  OpKind op = OpKind::Input;
  std::size_t self = 0;     // this node's SSA slot (== its index)
  std::size_t a = 0, b = 0; // operand slots
  info leaf = ^^int;        // Const only: reflection of the literal expression
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
  return op == OpKind::Add || op == OpKind::Sub || op == OpKind::Mul ||
         op == OpKind::Div || op == OpKind::Lt || op == OpKind::Le ||
         op == OpKind::Gt || op == OpKind::Ge || op == OpKind::Eq ||
         op == OpKind::Ne || op == OpKind::And || op == OpKind::Or ||
         op == OpKind::Select || op == OpKind::Max || op == OpKind::Min ||
         op == OpKind::Matmul;
}
consteval bool op_has_cond(OpKind op) { return op == OpKind::Select; }

// Ops valued 0/1. Derivative identically zero, so never varied.
consteval bool op_is_boolean(OpKind op) {
  return op == OpKind::Lt || op == OpKind::Le || op == OpKind::Gt ||
         op == OpKind::Ge || op == OpKind::Eq || op == OpKind::Ne ||
         op == OpKind::And || op == OpKind::Or || op == OpKind::Not;
}

// The arithmetic and functions primal() evaluates with: the built-in operators
// and <cmath>. A sweep with other needs (e.g. staying inside constant
// evaluation) passes its own type with the same members.
//
// primal() and these members are always inlined, so even an unoptimized
// build makes no call per node. (Unoptimized, the shared rules still cost
// forward_derivative ~10% over hand-written per-sweep tables, from parameter
// copies only optimization removes; at -O2 and up the cost is nil.)
struct BuiltinMath {
  template <typename T>
  [[gnu::always_inline]] static constexpr T add(T a, T b) {
    return a + b;
  }
  template <typename T>
  [[gnu::always_inline]] static constexpr T sub(T a, T b) {
    return a - b;
  }
  template <typename T>
  [[gnu::always_inline]] static constexpr T mul(T a, T b) {
    return a * b;
  }
  template <typename T>
  [[gnu::always_inline]] static constexpr T div(T a, T b) {
    return a / b;
  }
  template <typename T> [[gnu::always_inline]] static constexpr T neg(T a) {
    return -a;
  }
  template <typename T>
  [[gnu::always_inline]] static constexpr T max(T a, T b) {
    return (a < b) ? b : a;
  }
  template <typename T>
  [[gnu::always_inline]] static constexpr T min(T a, T b) {
    return (b < a) ? b : a;
  }
  template <OpKind Op, typename T>
  [[gnu::always_inline]] static constexpr T unary(T x) {
    if constexpr (Op == OpKind::Sin)
      return std::sin(x);
    else if constexpr (Op == OpKind::Cos)
      return std::cos(x);
    else if constexpr (Op == OpKind::Exp)
      return std::exp(x);
    else if constexpr (Op == OpKind::Log)
      return std::log(x);
    else if constexpr (Op == OpKind::Sqrt)
      return std::sqrt(x);
    else
      return std::erfc(x);
  }
};

// The ops primal() evaluates: every scalar op but the leaves, which each sweep
// reads from its own arguments (Input) or splices (Const).
consteval bool op_has_primal(OpKind op) {
  return op != OpKind::Input && op != OpKind::Const && op != OpKind::Matmul &&
         op != OpKind::Transpose && op != OpKind::Sum && op != OpKind::Relu;
}

// A node's value from the values of its operands a, b and (Select only) its
// condition c: the primal rule every sweep shares. The operands are passed by
// value, so all three slots are read, including a Select's untaken branch and
// an And / Or's b when a decides it, which may be behind a false guard and
// never written: a sweep must initialise every slot (each zero-fills `val`).
// Only the result ignores them -- Select's depends on the branch it takes, and
// And / Or's on b only when a leaves them undecided.
template <OpKind Op, typename Math = BuiltinMath, typename T>
[[gnu::always_inline]] constexpr T primal(T a, T b, T c) {
  if constexpr (Op == OpKind::Output)
    return a;
  else if constexpr (Op == OpKind::Add)
    return Math::add(a, b);
  else if constexpr (Op == OpKind::Sub)
    return Math::sub(a, b);
  else if constexpr (Op == OpKind::Mul)
    return Math::mul(a, b);
  else if constexpr (Op == OpKind::Div)
    return Math::div(a, b);
  else if constexpr (Op == OpKind::Neg)
    return Math::neg(a);
  else if constexpr (Op == OpKind::Sin || Op == OpKind::Cos ||
                     Op == OpKind::Exp || Op == OpKind::Log ||
                     Op == OpKind::Sqrt || Op == OpKind::Erfc)
    return Math::template unary<Op>(a);
  else if constexpr (Op == OpKind::Lt)
    return (a < b) ? T{1} : T{0};
  else if constexpr (Op == OpKind::Le)
    return (a <= b) ? T{1} : T{0};
  else if constexpr (Op == OpKind::Gt)
    return (a > b) ? T{1} : T{0};
  else if constexpr (Op == OpKind::Ge)
    return (a >= b) ? T{1} : T{0};
  else if constexpr (Op == OpKind::Eq)
    return (a == b) ? T{1} : T{0};
  else if constexpr (Op == OpKind::Ne)
    return (a != b) ? T{1} : T{0};
  else if constexpr (Op == OpKind::Not)
    return (a != T{0}) ? T{0} : T{1};
  else if constexpr (Op == OpKind::And)
    return (a != T{0} && b != T{0}) ? T{1} : T{0};
  else if constexpr (Op == OpKind::Or)
    return (a != T{0} || b != T{0}) ? T{1} : T{0};
  else if constexpr (Op == OpKind::Select)
    return (c != T{0}) ? a : b;
  else if constexpr (Op == OpKind::Abs)
    return (a < T{0}) ? Math::neg(a) : a;
  else if constexpr (Op == OpKind::Max)
    return Math::max(a, b);
  else if constexpr (Op == OpKind::Min)
    return Math::min(a, b);
  else
    static_assert(false, "primal: not a scalar op (see op_has_primal)");
}

namespace detail {

struct Ctx {
  std::vector<Node> nodes;
  std::vector<info> envDecl; // decl -> slot environment
  std::vector<std::size_t> envSlot;
  std::vector<info> callStack; // callees currently being inlined (cycle guard)
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

// Forward decls: lower / lower_body / inline_call are mutually recursive (a
// call site inlines its callee's body, which may contain further calls).
consteval std::size_t lower(Ctx &c, info e);
consteval std::size_t lower_body(Ctx &c, info body);
consteval std::size_t inline_call(Ctx &c, info call);

// Resolve a referenced variable to its SSA slot by identifier name. We match by
// name (unique within a straight-line body) rather than reflection identity
// because parameters_of yields ReflectionKind::Parameter while a decl_ref's
// declaration_of yields ReflectionKind::Declaration — different kinds that
// never compare equal. Last-wins so a local shadowing an outer name resolves to
// the most recent binding. An unresolved name (e.g. a global) has no slot;
// erroring beats returning a sentinel that would index the node array out of
// bounds.
consteval std::size_t findSlot(const Ctx &c, std::string_view name) {
  for (std::size_t i = c.envDecl.size(); i-- > 0;)
    if (m::identifier_of(c.envDecl[i]) == name)
      return c.envSlot[i];
  throw "reflection AD: name is not a parameter or local of the reflected body";
}

// Erroring beats a plausible default: falling back to Add would turn `x < y`
// into `x + y` -- a wrong derivative with no diagnostic.
consteval OpKind binOp(m::operators op) {
  if (op == m::operators::op_plus)
    return OpKind::Add;
  if (op == m::operators::op_minus)
    return OpKind::Sub;
  if (op == m::operators::op_star)
    return OpKind::Mul;
  if (op == m::operators::op_slash)
    return OpKind::Div;
  if (op == m::operators::op_less)
    return OpKind::Lt;
  if (op == m::operators::op_less_equals)
    return OpKind::Le;
  if (op == m::operators::op_greater)
    return OpKind::Gt;
  if (op == m::operators::op_greater_equals)
    return OpKind::Ge;
  if (op == m::operators::op_equals_equals)
    return OpKind::Eq;
  if (op == m::operators::op_exclamation_equals)
    return OpKind::Ne;
  if (op == m::operators::op_ampersand_ampersand)
    return OpKind::And;
  if (op == m::operators::op_pipe_pipe)
    return OpKind::Or;
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
template <class T> inline T p_sin(T x) { return std::sin(x); }
template <class T> inline T p_cos(T x) { return std::cos(x); }
template <class T> inline T p_exp(T x) { return std::exp(x); }
template <class T> inline T p_log(T x) { return std::log(x); }
template <class T> inline T p_sqrt(T x) { return std::sqrt(x); }
template <class T> inline T p_erfc(T x) { return std::erfc(x); }
// The kinks. Real ops rather than desugared Selects, so `is_continuous_on`
// can see that |x| is continuous across zero. Max/Min follow std::max
// (`a < b ? b : a`), so a tie yields the first operand.
template <class T> inline T p_fabs(T x) { return std::fabs(x); }
template <class T> inline T p_abs(T x) { return std::abs(x); }
template <class T> inline T p_fmax(T x, T y) { return std::fmax(x, y); }
template <class T> inline T p_fmin(T x, T y) { return std::fmin(x, y); }
template <class T> inline T p_max(T x, T y) { return std::max(x, y); }
template <class T> inline T p_min(T x, T y) { return std::min(x, y); }

struct Prim {
  bool found;
  OpKind op;
};

// Does `callee` match the float / double / long double instantiations of a
// probe template?
consteval bool is_std_fn(info callee, info pf, info pd, info pl) {
  return callee == probe_callee(pf) || callee == probe_callee(pd) ||
         callee == probe_callee(pl);
}

// Is `callee` a primitive? Checks the built-in std math set, then the
// user-extensible ad::primitive registry.
consteval Prim find_primitive(info callee) {
  if (is_std_fn(callee, ^^p_sin<float>, ^^p_sin<double>, ^^p_sin<long double>))
    return {true, OpKind::Sin};
  if (is_std_fn(callee, ^^p_cos<float>, ^^p_cos<double>, ^^p_cos<long double>))
    return {true, OpKind::Cos};
  if (is_std_fn(callee, ^^p_exp<float>, ^^p_exp<double>, ^^p_exp<long double>))
    return {true, OpKind::Exp};
  if (is_std_fn(callee, ^^p_log<float>, ^^p_log<double>, ^^p_log<long double>))
    return {true, OpKind::Log};
  if (is_std_fn(callee, ^^p_sqrt<float>, ^^p_sqrt<double>,
                ^^p_sqrt<long double>))
    return {true, OpKind::Sqrt};
  if (is_std_fn(callee, ^^p_erfc<float>, ^^p_erfc<double>,
                ^^p_erfc<long double>))
    return {true, OpKind::Erfc};
  if (is_std_fn(callee, ^^p_fabs<float>, ^^p_fabs<double>,
                ^^p_fabs<long double>) ||
      is_std_fn(callee, ^^p_abs<float>, ^^p_abs<double>, ^^p_abs<long double>))
    return {true, OpKind::Abs};
  if (is_std_fn(callee, ^^p_fmax<float>, ^^p_fmax<double>,
                ^^p_fmax<long double>) ||
      is_std_fn(callee, ^^p_max<float>, ^^p_max<double>, ^^p_max<long double>))
    return {true, OpKind::Max};
  if (is_std_fn(callee, ^^p_fmin<float>, ^^p_fmin<double>,
                ^^p_fmin<long double>) ||
      is_std_fn(callee, ^^p_min<float>, ^^p_min<double>, ^^p_min<long double>))
    return {true, OpKind::Min};

  info spec = m::substitute(^^primitive, {
                                             m::reflect_constant(callee)});
  if (m::is_complete_type(spec))
    for (info mem : m::members_of(spec, m::access_context::current()))
      if (m::has_identifier(mem) && m::identifier_of(mem) == "op")
        return {true, m::extract<OpKind>(mem)};
  return {false, OpKind::Input};
}

// Lower an expression to SSA nodes, returning its result slot.
consteval std::size_t lower(Ctx &c, info e) {
  e = stripCasts(e);

  if (m::is_variable_reference(e)) {
    const std::string_view name = m::identifier_of(m::declaration_of(e));
    for (std::size_t i = c.envDecl.size(); i-- > 0;)
      if (m::identifier_of(c.envDecl[i]) == name)
        return c.envSlot[i];

    // External constexpr variables (e.g. std::numbers constants) are not in
    // the local environment; fold them to literal constants in the DAG.
    return emit(c, OpKind::Const, 0, 0, m::constant_of(e));
  }

  if (m::is_literal(e)) {
    // Reduce the literal to a value reflection (constant_of); it is spliced by
    // the existing P2996 value splice at codegen -- no expression splicing.
    return emit(c, OpKind::Const, 0, 0, m::constant_of(e));
  }

  if (m::is_unary_operator(e)) {
    m::operators op = m::expression_operator_of(e);
    std::size_t a = lower(c, m::operands_of(e).front());
    if (op == m::operators::op_plus)
      return a; // unary plus is a no-op
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

#if defined(__clang__)
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
#endif // __clang__

  if (m::is_function_call(e)) {
    // operands_of(call) = [callee, arg0, arg1, ...].
    auto ops = m::operands_of(e);
    Prim p = find_primitive(m::callee_of(e));
    if (!p.found) // user helper -> inline its body
      return inline_call(c, e);
    OpKind op = p.op;
    if (op_has_b(op)) { // binary call, e.g. matmul(a, b)
      std::size_t a = lower(c, ops[1]);
      std::size_t b = lower(c, ops[2]);
      return emit(c, op, a, b);
    }
    std::size_t a = lower(c, ops[ops.size() - 1]); // unary call
    return emit(c, op, a);
  }

  // Unsupported: emit a constant placeholder from its value.
  return emit(c, OpKind::Const, 0, 0, m::constant_of(e));
}

// Lower the statements of a straight-line body into `c`, resolving names
// against the current environment; returns the slot of the (single) return
// value. Shared by build_nodes (top-level function) and inline_call (inlined
// callee). Anything outside that shape (control flow, assignment, a second
// return, no return at all) is rejected: skipping it would silently build a DAG
// that does not match the source, i.e. a wrong derivative with no diagnostic.
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
        throw "reflection AD: multiple return statements (straight-line bodies "
              "only)";
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
// analysis and codegen are unaffected, and arguments are evaluated once (a
// param used N times reuses its one slot).
consteval std::size_t inline_call(Ctx &c, info call) {
  auto ops = m::operands_of(call); // [callee, arg0, arg1, ...]
  info callee = m::callee_of(call);

  // Cycle guard: recursive helpers are out of scope (would not terminate here).
  for (info f : c.callStack)
    if (f == callee)
      throw "reflection AD: cannot inline a recursive function call";

  // A helper with no reflectable body (declaration only) cannot be inlined; a
  // silent constant would give a wrong derivative, so reject it explicitly.
  info body = m::body_of(callee);
  if (m::statements_of(body).empty())
    throw "reflection AD: cannot inline a call whose callee has no visible "
          "body";

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

  // (c) lower the callee body into the same DAG (nested calls inline via
  // lower).
  c.callStack.push_back(callee);
  std::size_t ret = lower_body(c, body);
  c.callStack.pop_back();

  // (d) close the scope: drop callee params + locals, restoring the caller env.
  c.envDecl.resize(mark);
  c.envSlot.resize(mark);
  return ret;
}

} // namespace detail

// Build the SSA node list for a reflected function. The last node is the root
// (an Output node forwarding the return value).
template <info Fn> consteval std::vector<Node> build_nodes() {
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
consteval bool
deriv_reads_operand_vals(OpKind op) { // reads val[a] (and val[b])
  return op == OpKind::Mul || op == OpKind::Div || op == OpKind::Sin ||
         op == OpKind::Cos || op == OpKind::Log || op == OpKind::Erfc ||
         // pick an operand's tangent at runtime, so read the operands
         op == OpKind::Abs || op == OpKind::Max || op == OpKind::Min;
}
consteval bool deriv_reads_self_val(OpKind op) { // reads val[self]
  return op == OpKind::Exp || op == OpKind::Sqrt;
}

// `active_mask` bit i set => input i is differentiated w.r.t. (~0ull = all).
consteval void mark_activity(std::vector<Node> &ns,
                             unsigned long long active_mask) {
  const std::size_t N = ns.size();

  // 1. varied: forward reachability from the differentiated input(s).
  std::vector<char> v(N, 0);
  for (Node &n : ns) {
    bool varied;
    if (n.op == OpKind::Input)
      varied = (active_mask >> n.self) & 1ull;
    else if (n.op == OpKind::Const || op_is_boolean(n.op))
      varied = false; // a predicate is piecewise constant: zero derivative
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
      if (op_has_a(n.op))
        need[n.a] = 1;
      if (op_has_b(n.op))
        need[n.b] = 1;
    }
    if (deriv_reads_self_val(n.op))
      need[n.self] = 1;
    // A Select's derivative reads the condition even when its value is dead.
    if (op_has_cond(n.op))
      need[n.cond] = 1;
  }
  for (std::size_t i = N; i-- > 0;) { // backward: needed val pulls in operands
    if (!need[i])
      continue;
    const Node &n = ns[i];
    if (op_has_a(n.op))
      need[n.a] = 1;
    if (op_has_b(n.op))
      need[n.b] = 1;
    if (op_has_cond(n.op))
      need[n.cond] = 1;
    if (n.guard != UNGUARDED)
      need[n.guard] = 1;
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
// How each node depends on one input, the target: shared by the analyses that
// solve for it (discontinuity_analysis, is_invertible).
// ---------------------------------------------------------------------------

// The reflected DAG of Fn, built once and shared by every analysis of Fn.
template <info Fn>
inline constexpr auto nodes_of = std::define_static_array(build_nodes<Fn>());

// Fn's number of arguments.
template <info Fn> consteval std::size_t input_count_of() {
  std::size_t count = 0;
  for (const Node &n : nodes_of<Fn>)
    count += n.op == OpKind::Input;
  return count;
}

struct Dependence {
  bool varies = false;  // it changes with the target
  bool affine = true;   // ... and only as c0 + c1 * target
  bool stepwise = true; // ... and only in steps, where a comparison flips
};

// A value that changes with the target other than in steps.
constexpr bool varies_continuously(Dependence d) {
  return d.varies && !d.stepwise;
}

// Per node, built in one forward pass (operands precede their users). Affine
// means built from the target with + -, unary - and * / by target-free
// values, and `?:` on a target-free condition. Stepwise means piecewise
// constant: every comparison or logical op is, and so is anything built only
// from stepwise and target-free values.
template <info Fn, std::size_t Target>
consteval std::vector<Dependence> target_dependence() {
  const auto nodes = nodes_of<Fn>;
  std::vector<Dependence> dep(nodes.size());
  for (const Node &n : nodes) {
    Dependence &d = dep[n.self];
    if (n.op == OpKind::Input) {
      d.varies = n.self == Target;
      d.stepwise = !d.varies;
      continue;
    }
    const Dependence a = op_has_a(n.op) ? dep[n.a] : Dependence{};
    const Dependence b = op_has_b(n.op) ? dep[n.b] : Dependence{};
    d.varies =
        a.varies || b.varies || (op_has_cond(n.op) && dep[n.cond].varies);
    if (n.op == OpKind::Add || n.op == OpKind::Sub)
      d.affine = a.affine && b.affine;
    else if (n.op == OpKind::Neg || n.op == OpKind::Output)
      d.affine = a.affine;
    else if (n.op == OpKind::Mul)
      d.affine = a.affine && b.affine && !(a.varies && b.varies);
    else if (n.op == OpKind::Div)
      d.affine = a.affine && !b.varies;
    else if (n.op == OpKind::Select && !dep[n.cond].varies)
      d.affine = a.affine && b.affine; // one branch, whatever the target
    else
      d.affine = !d.varies;
    // A Select's condition only picks a branch: if both branches are
    // stepwise, so is the Select.
    d.stepwise = op_is_boolean(n.op) ||
                 (!varies_continuously(a) && !varies_continuously(b));
  }
  return dep;
}

template <info Fn, std::size_t Target>
inline constexpr auto target_dependence_of =
    std::define_static_array(target_dependence<Fn, Target>());

} // namespace ad

#endif // REFLECT_DEMO_AUTOGRAD_H
