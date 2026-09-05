#ifndef REFLECT_DEMO_FORWARD_DERIVATIVE_H
#define REFLECT_DEMO_FORWARD_DERIVATIVE_H

#include <meta>

#include "autograd.h"

namespace ad {

using std::meta::info;

// Forward-mode directional derivative of `Fn` w.r.t. input index `Wrt`,
// evaluated at `args`. Compiles to inlined arithmetic (zero-cost).
template <std::meta::info Fn, std::size_t Wrt, typename T = double,
          typename... Args>
constexpr T forward_derivative(Args... args) {
  static constexpr auto nodes =
      std::define_static_array(build_marked_nodes<Fn, (1ull << Wrt)>());
  constexpr std::size_t N = nodes.size();

  const T in[] = {static_cast<T>(args)...};
  T val[N] = {};
  T tang[N] = {};

  template for (constexpr auto n : nodes) {
    // Primal (only where the value is actually read by a derivative).
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
        val[n.self] = (val[n.a] < val[n.b]) ? T{1} : T{0};
      else if constexpr (n.op == OpKind::Le)
        val[n.self] = (val[n.a] <= val[n.b]) ? T{1} : T{0};
      else if constexpr (n.op == OpKind::Gt)
        val[n.self] = (val[n.a] > val[n.b]) ? T{1} : T{0};
      else if constexpr (n.op == OpKind::Ge)
        val[n.self] = (val[n.a] >= val[n.b]) ? T{1} : T{0};
      else if constexpr (n.op == OpKind::Eq)
        val[n.self] = (val[n.a] == val[n.b]) ? T{1} : T{0};
      else if constexpr (n.op == OpKind::Ne)
        val[n.self] = (val[n.a] != val[n.b]) ? T{1} : T{0};
      else if constexpr (n.op == OpKind::Not)
        val[n.self] = (val[n.a] != T{0}) ? T{0} : T{1};
      // Reads val[b] only when val[a] leaves it undecided -- exactly when the
      // right operand was lowered as reachable, so its slot is written.
      else if constexpr (n.op == OpKind::And)
        val[n.self] = (val[n.a] != T{0} && val[n.b] != T{0}) ? T{1} : T{0};
      else if constexpr (n.op == OpKind::Or)
        val[n.self] = (val[n.a] != T{0} || val[n.b] != T{0}) ? T{1} : T{0};
      // Select likewise reads only the branch it takes.
      else if constexpr (n.op == OpKind::Select)
        val[n.self] = (val[n.cond] != T{0}) ? val[n.a] : val[n.b];
      else if constexpr (n.op == OpKind::Abs)
        val[n.self] = (val[n.a] < T{0}) ? -val[n.a] : val[n.a];
      else if constexpr (n.op == OpKind::Max)
        val[n.self] = (val[n.a] < val[n.b]) ? val[n.b] : val[n.a];
      else if constexpr (n.op == OpKind::Min)
        val[n.self] = (val[n.b] < val[n.a]) ? val[n.b] : val[n.a];
    }

    // Tangent (activity-gated: a non-varied operand contributes a zero term,
    // which is dropped instead of emitted as `... * 0`).
    if constexpr (n.op == OpKind::Input) {
      tang[n.self] = (n.self == Wrt) ? T{1} : T{0};
    } else if constexpr (!n.vself) {
      tang[n.self] = T{0}; // not varied (includes Const and every predicate)
    } else {
      if constexpr (n.guard != UNGUARDED) {
        if (!(val[n.guard] != T{0}))
          continue;
      }
      if constexpr (n.op == OpKind::Output)
        tang[n.self] = tang[n.a];
      else if constexpr (n.op == OpKind::Neg)
        tang[n.self] = -tang[n.a];
      else if constexpr (n.op == OpKind::Sin)
        tang[n.self] = std::cos(val[n.a]) * tang[n.a];
      else if constexpr (n.op == OpKind::Cos)
        tang[n.self] = -std::sin(val[n.a]) * tang[n.a];
      else if constexpr (n.op == OpKind::Exp)
        tang[n.self] = val[n.self] * tang[n.a];
      else if constexpr (n.op == OpKind::Log)
        tang[n.self] = tang[n.a] / val[n.a];
      else if constexpr (n.op == OpKind::Sqrt)
        tang[n.self] = tang[n.a] / (T{2} * val[n.self]);
      else if constexpr (n.op == OpKind::Erfc)
        tang[n.self] = tang[n.a] * -1 * two_over_root_pi *
                       (std::exp(-1 * (val[n.a] * val[n.a])));
      else if constexpr (n.op == OpKind::Add) {
        if constexpr (n.va && n.vb)
          tang[n.self] = tang[n.a] + tang[n.b];
        else if constexpr (n.va)
          tang[n.self] = tang[n.a];
        else
          tang[n.self] = tang[n.b];
      } else if constexpr (n.op == OpKind::Sub) {
        if constexpr (n.va && n.vb)
          tang[n.self] = tang[n.a] - tang[n.b];
        else if constexpr (n.va)
          tang[n.self] = tang[n.a];
        else
          tang[n.self] = -tang[n.b];
      } else if constexpr (n.op == OpKind::Mul) {
        if constexpr (n.va && n.vb)
          tang[n.self] = tang[n.a] * val[n.b] + val[n.a] * tang[n.b];
        else if constexpr (n.va)
          tang[n.self] = tang[n.a] * val[n.b];
        else
          tang[n.self] = val[n.a] * tang[n.b];
      } else if constexpr (n.op == OpKind::Div) {
        if constexpr (n.va && n.vb)
          tang[n.self] = (tang[n.a] * val[n.b] - val[n.a] * tang[n.b]) /
                         (val[n.b] * val[n.b]);
        else if constexpr (n.va)
          tang[n.self] = tang[n.a] / val[n.b];
        else
          tang[n.self] = -val[n.a] * tang[n.b] / (val[n.b] * val[n.b]);
      } else if constexpr (n.op == OpKind::Select) {
        if constexpr (n.va && n.vb)
          tang[n.self] = (val[n.cond] != T{0}) ? tang[n.a] : tang[n.b];
        else if constexpr (n.va)
          tang[n.self] = (val[n.cond] != T{0}) ? tang[n.a] : T{0};
        else
          tang[n.self] = (val[n.cond] != T{0}) ? T{0} : tang[n.b];
      } else if constexpr (n.op == OpKind::Abs) {
        tang[n.self] = (val[n.a] < T{0}) ? -tang[n.a] : tang[n.a];
      } else if constexpr (n.op == OpKind::Max || n.op == OpKind::Min) {
        // Both pass one operand through; they differ only in which way round.
        const bool takes_b = (n.op == OpKind::Max) ? (val[n.a] < val[n.b])
                                                   : (val[n.b] < val[n.a]);
        if constexpr (n.va && n.vb)
          tang[n.self] = takes_b ? tang[n.b] : tang[n.a];
        else if constexpr (n.va)
          tang[n.self] = takes_b ? T{0} : tang[n.a];
        else
          tang[n.self] = takes_b ? tang[n.b] : T{0};
      } else {
        tang[n.self] = T{0}; // any unhandled op
      }
    }
  }

  return tang[N - 1];
}

namespace detail {
template <std::meta::info Fn, typename T, typename... Args, std::size_t... I>
constexpr std::array<T, sizeof...(I)> grad_impl(std::index_sequence<I...>,
                                                Args... args) {
  return {forward_derivative<Fn, I, T>(args...)...};
}
} // namespace detail

// Full gradient via forward mode: one forward pass per input (P passes).
template <std::meta::info Fn, typename T = double, typename... Args>
constexpr std::array<T, sizeof...(Args)> gradient_of(Args... args) {
  return detail::grad_impl<Fn, T>(std::index_sequence_for<Args...>{}, args...);
}

} // namespace ad

#endif // REFLECT_DEMO_FORWARD_DERIVATIVE_H
