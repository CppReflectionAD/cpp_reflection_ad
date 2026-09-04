#ifndef REFLECT_DEMO_RECURSIVE_HIGHER_ORDER_DERIVATIVE_H
#define REFLECT_DEMO_RECURSIVE_HIGHER_ORDER_DERIVATIVE_H

#include <meta>

#include "autograd.h"

namespace ad {

using std::meta::info;

// ---------------------------------------------------------------------------
// Higher-order derivatives by recursive DAG differentiation.
// ---------------------------------------------------------------------------
namespace detail {

// Raw append: always pushes a new node and returns its slot.
consteval std::size_t emit_raw(std::vector<Node> &out, OpKind op, std::size_t a,
                               std::size_t b, info leaf = ^^int,
                               std::size_t cond = 0,
                               std::size_t guard = UNGUARDED) {
  std::size_t s = out.size();
  out.push_back(Node{op, s, a, b, leaf, cond, guard});
  return s;
}

// Reuse a node with the same op and operands rather than bloating the DAG.
// `cond` and `guard` are in the key: a value computed under one guard must not
// be reused under another, where its slot may never have been written.
consteval std::size_t emit_node(std::vector<Node> &out, OpKind op,
                                std::size_t a, std::size_t b, info leaf = ^^int,
                                std::size_t cond = 0,
                                std::size_t guard = UNGUARDED) {
  for (const Node &n : out)
    if (n.op == op && n.a == a && n.b == b && n.cond == cond &&
        n.guard == guard)
      return n.self;
  return emit_raw(out, op, a, b, leaf, cond, guard);
}

// A compile-time map from a constant scalar value to the slot of its (unique)
// Const node.
using ConstPool = std::vector<std::pair<double, std::size_t>>;

// Return the slot of the Const node holding `value`, creating it (and recording
// it in `pool`) on first request. `emit_node` caches based on operands, but
// `ConstPool` caches based on the leaf value.
consteval std::size_t ensure_const_node(std::vector<Node> &out, ConstPool &pool,
                                        double value) {
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
  std::vector<std::size_t> tang(
      M, 0); // tang[i] returns the slot in `out` that corresponds to the
             // tangent of the ith node in `src`.
  std::vector<bool> varied(M, 0); // varied[i] marks whether the tangent of ith
                                  // node in 'src' is non-zero.

  // Copy every primal node except the Output (which is at the end of src).
  // Copying in order keeps each node at its original slot, so `src` Node
  // operands are not invalidated. We assume these will be reused in the
  // calculate for the derivative, and if not they will be pruned later.
  for (const Node &n : src)
    if (n.op != OpKind::Output)
      emit_raw(out, n.op, n.a, n.b, n.leaf, n.cond, n.guard);

  ConstPool constPool;

  // Create a tangent node for each `src` node in order.
  for (const Node &n : src) {
    const std::size_t i = n.self;
    // `va`/`vb` correspond to whether operand a/b has a nonzero tangent
    const bool va = op_has_a(n.op) && varied[n.a];
    const bool vb = op_has_b(n.op) && varied[n.b];
    // A derivative node inherits the guard of the primal it came from. Consts
    // are the exception -- safe anywhere, so they stay unguarded.
    auto emit_here = [&](OpKind op, std::size_t x, std::size_t y) {
      return emit_node(out, op, x, y, ^^int, 0, n.guard);
    };
    switch (n.op) {
    case OpKind::Input:
      varied[i] = (i == wrt);
      if (varied[i])
        tang[i] = ensure_const_node(out, constPool, 1.0);
      break;
    case OpKind::Const:
      varied[i] = false;
      break;
    case OpKind::Output: // forwards the return value's tangent
      varied[i] = va;
      if (va)
        tang[i] = tang[n.a];
      break;
    case OpKind::Add:
      varied[i] = va || vb;
      if (va && vb)
        tang[i] = emit_here(OpKind::Add, tang[n.a], tang[n.b]);
      else if (va)
        tang[i] = tang[n.a];
      else if (vb)
        tang[i] = tang[n.b];
      break;
    case OpKind::Sub:
      varied[i] = va || vb;
      if (va && vb)
        tang[i] = emit_here(OpKind::Sub, tang[n.a], tang[n.b]);
      else if (va)
        tang[i] = tang[n.a];
      else if (vb)
        tang[i] = emit_here(OpKind::Neg, tang[n.b], 0);
      break;
    case OpKind::Mul: // d(ab) = da*b + a*db
      varied[i] = va || vb;
      if (va && vb) {
        // product rule
        std::size_t l = emit_here(OpKind::Mul, tang[n.a], n.b);
        std::size_t r = emit_here(OpKind::Mul, n.a, tang[n.b]);
        tang[i] = emit_here(OpKind::Add, l, r);
      } else if (va)
        tang[i] = emit_here(OpKind::Mul, tang[n.a], n.b);
      else if (vb)
        tang[i] = emit_here(OpKind::Mul, n.a, tang[n.b]);
      break;
    case OpKind::Div: // d(a/b) = (da*b - a*db) / (b*b)
      varied[i] = va || vb;
      if (va && vb) {
        // quotient rule
        std::size_t l = emit_here(OpKind::Mul, tang[n.a], n.b);
        std::size_t r = emit_here(OpKind::Mul, n.a, tang[n.b]);
        std::size_t numerator = emit_here(OpKind::Sub, l, r);
        std::size_t denominator = emit_here(OpKind::Mul, n.b, n.b);
        tang[i] = emit_here(OpKind::Div, numerator, denominator);
      } else if (va) {
        tang[i] = emit_here(OpKind::Div, tang[n.a], n.b);
      } else if (vb) { // -a*db / (b*b)
        std::size_t numerator = emit_here(OpKind::Mul, n.a, tang[n.b]);
        std::size_t denominator = emit_here(OpKind::Mul, n.b, n.b);
        std::size_t quotient = emit_here(OpKind::Div, numerator, denominator);
        tang[i] = emit_here(OpKind::Neg, quotient, 0);
      }
      break;
    case OpKind::Neg:
      varied[i] = va;
      if (va)
        tang[i] = emit_here(OpKind::Neg, tang[n.a], 0);
      break;
    case OpKind::Sin: // cos(a) * da
      varied[i] = va;
      if (va) {
        std::size_t c = emit_here(OpKind::Cos, n.a, 0);
        tang[i] = emit_here(OpKind::Mul, c, tang[n.a]);
      }
      break;
    case OpKind::Cos: // -sin(a) * da
      varied[i] = va;
      if (va) {
        std::size_t s = emit_here(OpKind::Sin, n.a, 0);
        std::size_t p = emit_here(OpKind::Mul, s, tang[n.a]);
        tang[i] = emit_here(OpKind::Neg, p, 0);
      }
      break;
    case OpKind::Exp: // exp(a) * da
      varied[i] = va;
      // slot i is `exp(a)`
      if (va)
        tang[i] = emit_here(OpKind::Mul, i, tang[n.a]);
      break;
    case OpKind::Log: // da / a
      varied[i] = va;
      if (va)
        tang[i] = emit_here(OpKind::Div, tang[n.a], n.a);
      break;
    case OpKind::Sqrt: // da / (2*sqrt(a)) = da / (2 * self)
      varied[i] = va;
      if (va) {
        std::size_t two = ensure_const_node(out, constPool, 2.0);
        // slot i is `sqrt(a)`
        std::size_t two_sqrt_a = emit_here(OpKind::Mul, two, i);
        tang[i] = emit_here(OpKind::Div, tang[n.a], two_sqrt_a);
      }
      break;
    case OpKind::Erfc: // -2/sqrt(pi) * exp(-a*a) * da
      varied[i] = va;
      if (va) {
        std::size_t k = ensure_const_node(out, constPool, -two_over_root_pi);
        std::size_t a_squared = emit_here(OpKind::Mul, n.a, n.a);
        std::size_t minus_a_squared = emit_here(OpKind::Neg, a_squared, 0);
        std::size_t exp_minus_a_squared =
            emit_here(OpKind::Exp, minus_a_squared, 0);
        std::size_t ke = emit_here(OpKind::Mul, k, exp_minus_a_squared);
        tang[i] = emit_here(OpKind::Mul, ke, tang[n.a]);
      }
      break;
    case OpKind::Lt:
    case OpKind::Le:
    case OpKind::Gt:
    case OpKind::Ge:
    case OpKind::Eq:
    case OpKind::Ne:
    case OpKind::And:
    case OpKind::Or:
    case OpKind::Not:
      varied[i] = false; // a predicate is piecewise constant
      break;
    case OpKind::Select: // the tangent follows the branch taken
      varied[i] = va || vb;
      if (varied[i]) {
        std::size_t zero = ensure_const_node(out, constPool, 0.0);
        tang[i] = emit_node(out, OpKind::Select, va ? tang[n.a] : zero,
                            vb ? tang[n.b] : zero, ^^int, n.cond, n.guard);
      }
      break;
    case OpKind::Abs: // d|a| = (a < 0 ? -da : da)
      varied[i] = va;
      if (va) {
        std::size_t zero = ensure_const_node(out, constPool, 0.0);
        std::size_t isNeg = emit_here(OpKind::Lt, n.a, zero);
        std::size_t neg_tangent = emit_here(OpKind::Neg, tang[n.a], 0);
        tang[i] = emit_node(out, OpKind::Select, neg_tangent, tang[n.a], ^^int,
                            isNeg, n.guard);
      }
      break;
    case OpKind::Max: // max(a,b) = (a < b ? b : a)
    case OpKind::Min: // min(a,b) = (b < a ? b : a)
      varied[i] = va || vb;
      if (varied[i]) {
        std::size_t zero = ensure_const_node(out, constPool, 0.0);
        std::size_t cmp = (n.op == OpKind::Max)
                              ? emit_here(OpKind::Lt, n.a, n.b)
                              : emit_here(OpKind::Lt, n.b, n.a);
        tang[i] = emit_node(out, OpKind::Select, vb ? tang[n.b] : zero,
                            va ? tang[n.a] : zero, ^^int, cmp, n.guard);
      }
      break;
    default:
      throw "reflection AD: no derivative rule for this op";
      break;
    }
  }

  // Create a new output node.
  const std::size_t srcOutputIndex = M - 1;
  std::size_t srcOutputDerivative =
      varied[srcOutputIndex] ? tang[srcOutputIndex]
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
    if (!need[k])
      continue;
    const Node &n = ns[k];
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

} // namespace detail

// Create nodes for a DAG corresponding to the partial derivative of `Fn`,
// differentiated with respect to each argument in `Wrts...`. Creates a DAG for
// the original `Fn`, and then recursively creates subsequent DAGs for each
// derivative using forward AD.
template <info Fn, std::size_t... Wrts>
consteval std::vector<Node> build_partial_nodes() {
  std::vector<Node> ns = build_nodes<Fn>();
  const std::size_t list[] = {Wrts...};
  for (std::size_t w : list)
    ns = detail::differentiate(ns, w);
  detail::prune_reachable(ns);
  return ns;
}

// Partial derivative of `Fn` with respect to each index in `Wrts...`, evaluated
// at `Arg...`.
template <info Fn, std::size_t... Wrts, typename... Args>
constexpr double partial_derivative(Args... args) {
  static constexpr auto nodes =
      std::define_static_array(build_partial_nodes<Fn, Wrts...>());
  constexpr std::size_t N = nodes.size();
  const double in[] = {static_cast<double>(args)...};
  double val[N] = {};
  template for (constexpr auto n : nodes) {
    if constexpr (n.nself) {
      // A guarded node belongs to a branch; skip it unless that branch is
      // taken.
      if constexpr (n.guard != UNGUARDED) {
        if (!(val[n.guard] != 0.0))
          continue;
      }
      if constexpr (n.op == OpKind::Input)
        val[n.self] = in[n.self];
      else if constexpr (n.op == OpKind::Const)
        val[n.self] = static_cast<double>([:n.leaf:]);
      else if constexpr (n.op == OpKind::Output)
        val[n.self] = val[n.a];
      else if constexpr (n.op == OpKind::Add)
        val[n.self] = val[n.a] + val[n.b];
      else if constexpr (n.op == OpKind::Sub)
        val[n.self] = val[n.a] - val[n.b];
      else if constexpr (n.op == OpKind::Mul)
        val[n.self] = val[n.a] * val[n.b];
      else if constexpr (n.op == OpKind::Div)
        val[n.self] = val[n.a] / val[n.b];
      else if constexpr (n.op == OpKind::Neg)
        val[n.self] = -val[n.a];
      else if constexpr (n.op == OpKind::Sin)
        val[n.self] = std::sin(val[n.a]);
      else if constexpr (n.op == OpKind::Cos)
        val[n.self] = std::cos(val[n.a]);
      else if constexpr (n.op == OpKind::Exp)
        val[n.self] = std::exp(val[n.a]);
      else if constexpr (n.op == OpKind::Log)
        val[n.self] = std::log(val[n.a]);
      else if constexpr (n.op == OpKind::Sqrt)
        val[n.self] = std::sqrt(val[n.a]);
      else if constexpr (n.op == OpKind::Erfc)
        val[n.self] = std::erfc(val[n.a]);
      else if constexpr (n.op == OpKind::Lt)
        val[n.self] = (val[n.a] < val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Le)
        val[n.self] = (val[n.a] <= val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Gt)
        val[n.self] = (val[n.a] > val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Ge)
        val[n.self] = (val[n.a] >= val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Eq)
        val[n.self] = (val[n.a] == val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Ne)
        val[n.self] = (val[n.a] != val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Not)
        val[n.self] = (val[n.a] != 0.0) ? 0.0 : 1.0;
      // Reads val[b] only when val[a] leaves it undecided -- exactly when the
      // right operand was lowered as reachable, so its slot is written.
      else if constexpr (n.op == OpKind::And)
        val[n.self] = (val[n.a] != 0.0 && val[n.b] != 0.0) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Or)
        val[n.self] = (val[n.a] != 0.0 || val[n.b] != 0.0) ? 1.0 : 0.0;
      // Select likewise reads only the branch it takes.
      else if constexpr (n.op == OpKind::Select)
        val[n.self] = (val[n.cond] != 0.0) ? val[n.a] : val[n.b];
      else if constexpr (n.op == OpKind::Abs)
        val[n.self] = (val[n.a] < 0.0) ? -val[n.a] : val[n.a];
      else if constexpr (n.op == OpKind::Max)
        val[n.self] = (val[n.a] < val[n.b]) ? val[n.b] : val[n.a];
      else if constexpr (n.op == OpKind::Min)
        val[n.self] = (val[n.b] < val[n.a]) ? val[n.b] : val[n.a];
    }
  }
  return val[N - 1];
}

} // namespace ad

#endif // REFLECT_DEMO_RECURSIVE_HIGHER_ORDER_DERIVATIVE_H
