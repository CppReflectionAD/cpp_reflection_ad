// nn/cmath_rules.hpp — <cmath> functions as primitives, for the scalar code
// around the tensors.
//
// `^^std::exp` names an overload set, so these can't be vjp<key(^^f)>
// specializations. Each rule lists the float, double and long double
// overloads it covers, recovered from a probe that calls them, and gives
// dy/dx from the argument (d_x) or from the result (d_y). That choice is also
// the value the reverse sweep keeps (nn::ad::reads::args or ::result), so the
// rules need no `reads`. An integer argument selects <cmath>'s integer
// overload, which is not covered.

#ifndef NN_CMATH_RULES_HPP
#define NN_CMATH_RULES_HPP

#include "autodiff.hpp"

#include <cmath>
#include <span>

namespace nn::ad::cmath_probe {
template <class T> T exp(T x) { return std::exp(x); }
template <class T> T log(T x) { return std::log(x); }
template <class T> T sqrt(T x) { return std::sqrt(x); }
template <class T> T sin(T x) { return std::sin(x); }
template <class T> T cos(T x) { return std::cos(x); }
template <class T> T tanh(T x) { return std::tanh(x); }
} // namespace nn::ad::cmath_probe

namespace nn::ad::callee_rules {

struct exp {
  static constexpr std::span<const info> callees =
      probed({^^cmath_probe::exp<float>, ^^cmath_probe::exp<double>,
              ^^cmath_probe::exp<long double>});
  static constexpr auto d_y(auto y) { return y; }
};

struct log {
  static constexpr std::span<const info> callees =
      probed({^^cmath_probe::log<float>, ^^cmath_probe::log<double>,
              ^^cmath_probe::log<long double>});
  static constexpr auto d_x(auto x) { return 1 / x; }
};

struct sqrt {
  static constexpr std::span<const info> callees =
      probed({^^cmath_probe::sqrt<float>, ^^cmath_probe::sqrt<double>,
              ^^cmath_probe::sqrt<long double>});
  static constexpr auto d_y(auto y) { return 1 / (2 * y); }
};

struct sin {
  static constexpr std::span<const info> callees =
      probed({^^cmath_probe::sin<float>, ^^cmath_probe::sin<double>,
              ^^cmath_probe::sin<long double>});
  static auto d_x(auto x) { return std::cos(x); }
};

struct cos {
  static constexpr std::span<const info> callees =
      probed({^^cmath_probe::cos<float>, ^^cmath_probe::cos<double>,
              ^^cmath_probe::cos<long double>});
  static auto d_x(auto x) { return -std::sin(x); }
};

struct tanh {
  static constexpr std::span<const info> callees =
      probed({^^cmath_probe::tanh<float>, ^^cmath_probe::tanh<double>,
              ^^cmath_probe::tanh<long double>});
  static constexpr auto d_y(auto y) { return 1 - y * y; }
};

} // namespace nn::ad::callee_rules

#endif // NN_CMATH_RULES_HPP
