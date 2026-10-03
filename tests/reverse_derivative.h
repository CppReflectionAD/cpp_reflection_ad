#ifndef REFLECT_DEMO_REVERSE_DERIVATIVE_H
#define REFLECT_DEMO_REVERSE_DERIVATIVE_H

#include <meta>

#include "autograd.h"

namespace ad {

using std::meta::info;

// Full gradient via reverse mode: one primal sweep + one adjoint sweep over the
// reversed DAG, computing every partial in a single pass. This is the efficient
// path for scalar-output, many-input functions (the "autograd" case).
template <info Fn, typename T = double, typename... Args>
constexpr std::array<T, sizeof...(Args)> gradient_reverse(Args... args) {
  static constexpr auto nodes =
      std::define_static_array(build_marked_nodes<Fn, ~0ull>());
  static constexpr auto rnodes =
      std::define_static_array(build_marked_nodes_reversed<Fn, ~0ull>());
  constexpr std::size_t N = nodes.size();
  constexpr std::size_t P = sizeof...(Args);

  const T in[] = {static_cast<T>(args)...};
  T val[N] = {};
  T adj[N] = {}; // adjoints, accumulated; zero-initialized

  // Forward (primal) sweep: compute the values the adjoint rules will read.
  template for (constexpr auto n : nodes) {
    if constexpr (n.nself) {
      // A guarded node belongs to a branch; skip it unless that branch is
      // taken.
      if constexpr (n.guard != UNGUARDED) {
        if (!(val[n.guard] != T{0}))
          continue;
      }
      if constexpr (n.op == OpKind::Input)
        val[n.self] = in[n.self];
      else if constexpr (n.op == OpKind::Const)
        val[n.self] = static_cast<T>([:n.leaf:]);
      else if constexpr (op_has_primal(n.op))
        val[n.self] = primal<n.op>(val[n.a], val[n.b], val[n.cond]);
    }
  }

  // Seed the output adjoint, then sweep the DAG in reverse, pushing each node's
  // adjoint to its operands via the local VJP (accumulating with +=).
  adj[N - 1] = T{1};
  template for (constexpr auto n : rnodes) {
    if constexpr (n.va || n.vb) {
      if constexpr (n.guard != UNGUARDED) {
        if (!(val[n.guard] != T{0}))
          continue;
      }
      if constexpr (n.op == OpKind::Output) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self];
      } else if constexpr (n.op == OpKind::Add) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self];
        if constexpr (n.vb)
          adj[n.b] += adj[n.self];
      } else if constexpr (n.op == OpKind::Sub) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self];
        if constexpr (n.vb)
          adj[n.b] -= adj[n.self];
      } else if constexpr (n.op == OpKind::Mul) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self] * val[n.b];
        if constexpr (n.vb)
          adj[n.b] += adj[n.self] * val[n.a];
      } else if constexpr (n.op == OpKind::Div) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self] / val[n.b];
        if constexpr (n.vb)
          adj[n.b] -= adj[n.self] * val[n.a] / (val[n.b] * val[n.b]);
      } else if constexpr (n.op == OpKind::Neg) {
        if constexpr (n.va)
          adj[n.a] -= adj[n.self];
      } else if constexpr (n.op == OpKind::Sin) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self] * std::cos(val[n.a]);
      } else if constexpr (n.op == OpKind::Cos) {
        if constexpr (n.va)
          adj[n.a] += -adj[n.self] * std::sin(val[n.a]);
      } else if constexpr (n.op == OpKind::Exp) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self] * val[n.self];
      } else if constexpr (n.op == OpKind::Log) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self] / val[n.a];
      } else if constexpr (n.op == OpKind::Sqrt) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self] / (T{2} * val[n.self]);
      } else if constexpr (n.op == OpKind::Erfc) {
        if constexpr (n.va)
          adj[n.a] += adj[n.self] * -1 * two_over_root_pi *
                      (std::exp(-1 * (val[n.a] * val[n.a])));
      } else if constexpr (n.op == OpKind::Select) {
        // The adjoint flows only down the branch that was taken.
        if constexpr (n.va) {
          if (val[n.cond] != T{0})
            adj[n.a] += adj[n.self];
        }
        if constexpr (n.vb) {
          if (!(val[n.cond] != T{0}))
            adj[n.b] += adj[n.self];
        }
      } else if constexpr (n.op == OpKind::Abs) {
        if constexpr (n.va)
          adj[n.a] += (val[n.a] < T{0}) ? -adj[n.self] : adj[n.self];
      } else if constexpr (n.op == OpKind::Max || n.op == OpKind::Min) {
        const bool takes_b = (n.op == OpKind::Max) ? (val[n.a] < val[n.b])
                                                   : (val[n.b] < val[n.a]);
        if constexpr (n.va) {
          if (!takes_b)
            adj[n.a] += adj[n.self];
        }
        if constexpr (n.vb) {
          if (takes_b)
            adj[n.b] += adj[n.self];
        }
      }
    }
  }

  // Input k's accumulated adjoint is the k-th partial derivative.
  std::array<T, P> g{};
  for (std::size_t i = 0; i < P; ++i)
    g[i] = adj[i];
  return g;
}

} // namespace ad

#endif // REFLECT_DEMO_REVERSE_DERIVATIVE_H
