// autograd_tensor.h — reverse-mode AD over *tensor* code, via the same engine.
//
// Reuses the reflection IR from autograd.h (build_marked_nodes / activity
// analysis) and adds a reverse sweep whose VJP rules are written against a
// generic op *vocabulary* — matmul, transpose, sum, relu, elementwise
// add/sub/mul/div, and the helpers zeros_like / ones_like / unbroadcast /
// d_relu — all called *unqualified* so they resolve by ADL to whatever tensor
// type you use. The engine names no concrete tensor type, so it works with a
// custom Tensor, Eigen, libtorch, … given those ops.
//
// gradient_wrt<^^loss, T, P0, P1, ...>(inputs...) returns the gradients w.r.t.
// the input indices P0, P1, ... (the "parameters"). Non-listed inputs (e.g.
// data) are marked inactive, so activity analysis prunes their gradient work —
// exactly `requires_grad`.
//
// Assumptions for a clean demo: the reflected function is in A-normal form
// (one op per statement, operands are named variables) and uses *named* ops
// (add/mul/matmul/relu/sum/...), so the reflected AST stays simple. Output is a
// scalar (0-d tensor) loss.

#ifndef REFLECT_DEMO_AUTOGRAD_TENSOR_H
#define REFLECT_DEMO_AUTOGRAD_TENSOR_H

#include "../autograd.h"

namespace ad {

template <info Fn, typename T, std::size_t... Params, typename... Args>
std::array<T, sizeof...(Params)> gradient_wrt(const Args &...args) {
  constexpr unsigned long long mask = ((1ull << Params) | ... | 0ull);
  constexpr auto nodes  = std::define_static_array(build_marked_nodes<Fn, mask>());
  constexpr auto rnodes = std::define_static_array(build_marked_nodes_reversed<Fn, mask>());
  constexpr std::size_t N = nodes.size();

  // Inputs are referenced, not copied (args stay live for the call). Only the
  // one unavoidable copy into val[] for an Input node happens, in the sweep.
  const T *in[] = { (&args)... };
  T val[N];
  T adj[N];
  bool seen[N] = {};   // lazy adjoint init: first write assigns, later ones add

  // --- forward (primal) sweep: compute every node's value ---
  template for (constexpr auto n : nodes) {
    if constexpr (n.op == OpKind::Input)       val[n.self] = *in[n.self];
    else if constexpr (n.op == OpKind::Const)  val[n.self] = static_cast<T>([: n.leaf :]);
    else if constexpr (n.op == OpKind::Output) val[n.self] = val[n.a];
    else if constexpr (n.op == OpKind::Add)    val[n.self] = add(val[n.a], val[n.b]);
    else if constexpr (n.op == OpKind::Sub)    val[n.self] = sub(val[n.a], val[n.b]);
    else if constexpr (n.op == OpKind::Mul)    val[n.self] = mul(val[n.a], val[n.b]);
    else if constexpr (n.op == OpKind::Div)    val[n.self] = div(val[n.a], val[n.b]);
    else if constexpr (n.op == OpKind::Neg)    val[n.self] = neg(val[n.a]);
    else if constexpr (n.op == OpKind::Matmul) val[n.self] = matmul(val[n.a], val[n.b]);
    else if constexpr (n.op == OpKind::Transpose) val[n.self] = transpose(val[n.a]);
    else if constexpr (n.op == OpKind::Sum)    val[n.self] = sum(val[n.a]);
    else if constexpr (n.op == OpKind::Relu)   val[n.self] = relu(val[n.a]);
  }

  // adjoint accumulation with lazy init (can't `+=` into an unshaped zero).
  // Takes g by value so caller temporaries move in; the common first-write path
  // moves instead of copying.
  auto acc = [&](std::size_t i, T g) {
    if (seen[i]) adj[i] = add(adj[i], g);
    else { adj[i] = std::move(g); seen[i] = true; }
  };

  // seed: d(loss)/d(loss) = 1, shaped like the (scalar) output
  adj[N - 1] = ones_like(val[N - 1]);
  seen[N - 1] = true;

  // --- reverse (adjoint) sweep: push each node's adjoint to varied operands ---
  template for (constexpr auto n : rnodes) {
    if constexpr (n.op == OpKind::Output) {
      if constexpr (n.va) acc(n.a, adj[n.self]);
    } else if constexpr (n.op == OpKind::Add) {
      if constexpr (n.va) acc(n.a, unbroadcast(adj[n.self], val[n.a]));
      if constexpr (n.vb) acc(n.b, unbroadcast(adj[n.self], val[n.b]));
    } else if constexpr (n.op == OpKind::Sub) {
      if constexpr (n.va) acc(n.a, unbroadcast(adj[n.self], val[n.a]));
      if constexpr (n.vb) acc(n.b, unbroadcast(neg(adj[n.self]), val[n.b]));
    } else if constexpr (n.op == OpKind::Mul) {   // elementwise
      if constexpr (n.va) acc(n.a, unbroadcast(mul(adj[n.self], val[n.b]), val[n.a]));
      if constexpr (n.vb) acc(n.b, unbroadcast(mul(adj[n.self], val[n.a]), val[n.b]));
    } else if constexpr (n.op == OpKind::Div) {
      if constexpr (n.va) acc(n.a, unbroadcast(div(adj[n.self], val[n.b]), val[n.a]));
      if constexpr (n.vb) acc(n.b, unbroadcast(neg(div(mul(adj[n.self], val[n.a]),
                                                        mul(val[n.b], val[n.b]))), val[n.b]));
    } else if constexpr (n.op == OpKind::Neg) {
      if constexpr (n.va) acc(n.a, neg(adj[n.self]));
    } else if constexpr (n.op == OpKind::Matmul) {
      // C = A·B  =>  Ā += C̄·Bᵀ ,  B̄ += Aᵀ·C̄
      if constexpr (n.va) acc(n.a, matmul(adj[n.self], transpose(val[n.b])));
      if constexpr (n.vb) acc(n.b, matmul(transpose(val[n.a]), adj[n.self]));
    } else if constexpr (n.op == OpKind::Transpose) {
      if constexpr (n.va) acc(n.a, transpose(adj[n.self]));
    } else if constexpr (n.op == OpKind::Sum) {
      // scalar adjoint broadcast back to the summed tensor's shape
      if constexpr (n.va) acc(n.a, mul(ones_like(val[n.a]), adj[n.self]));
    } else if constexpr (n.op == OpKind::Relu) {
      if constexpr (n.va) acc(n.a, mul(adj[n.self], d_relu(val[n.a])));
    }
    // Input / Const: no operands to propagate to.
  }

  // gradient for each requested parameter (zeros if it never received one)
  return { (seen[Params] ? adj[Params] : zeros_like(val[Params]))... };
}

}  // namespace ad

#endif  // REFLECT_DEMO_AUTOGRAD_TENSOR_H
