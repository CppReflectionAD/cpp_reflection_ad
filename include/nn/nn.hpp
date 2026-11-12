// nn/nn.hpp — the whole library: tensor values and primitives, the reflection
// AD engine, and the rules connecting them. See README.md.

#ifndef NN_NN_HPP
#define NN_NN_HPP

#include "autodiff.hpp"
#include "cmath_rules.hpp"
#include "rules.hpp"
#include "tensor.hpp"

#include <cstddef>
#include <span>

namespace nn::ad {

// The tensor primitives and <cmath>. Put a rule first to override one:
// engine<my_rule, default_rules>.
using default_rules = rule_set<tensor_rule_set, cmath_rule_set>;
using default_engine = engine<default_rules>;

template <info Fn, unsigned long long Mask>
inline constexpr std::span<const Node> tape = default_engine::tape<Fn, Mask>;

template <info Fn, std::size_t... Wrt, class... Args>
auto value_and_grad(const Args &...args) {
  return default_engine::value_and_grad<Fn, Wrt...>(args...);
}

template <info Fn, std::size_t... Wrt, class... Args>
auto grad(const Args &...args) {
  return default_engine::grad<Fn, Wrt...>(args...);
}

} // namespace nn::ad

#endif // NN_NN_HPP
