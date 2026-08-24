#ifndef TEST_TOOLS_HPP
#define TEST_TOOLS_HPP

#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <type_traits>

namespace detail {

// Picks the float type with the fewest mantissa bits (lowest precision).
template <class T, class U>
using lower_precision_t = std::conditional_t<
    (std::numeric_limits<T>::digits <= std::numeric_limits<U>::digits), T, U>;

template <class D1, class D2, class Tol>
inline auto expect_near_abs(D1 val1, D2 val2, Tol tol)
    -> std::tuple<bool, lower_precision_t<lower_precision_t<D1, D2>, Tol>> {
  using L = lower_precision_t<lower_precision_t<D1, D2>, Tol>;
  const L v1 = static_cast<L>(val1);
  const L v2 = static_cast<L>(val2);
  const L t = static_cast<L>(tol);
  const L abs_diff = std::abs(v1 - v2);
  return std::make_tuple(abs_diff < std::abs(t), abs_diff);
}

template <class D1, class D2, class Tol>
inline auto expect_near_rel(D1 val1, D2 val2, Tol tol)
    -> std::tuple<bool, lower_precision_t<lower_precision_t<D1, D2>, Tol>> {

  using L = lower_precision_t<lower_precision_t<D1, D2>, Tol>;
  const L v1 = static_cast<L>(val1);
  const L v2 = static_cast<L>(val2);
  const L t = static_cast<L>(tol);

  // should take care of the val1=val2=0 case
  if (v1 == v2) {
    return std::make_tuple(true, 0.);
  }

  L average = (v1 + v2) * 0.5;

  if (average == 0.) {
    average = std::max(std::abs(v1), std::abs(v2));
  }

  L rel_diff = (v1 - v2) / average;

  return std::make_tuple(rel_diff < t, rel_diff);
}

} // namespace detail

static int _result = 0;

#define INCREASE                                                               \
  do {                                                                         \
    _result++;                                                                 \
    constexpr int limit = 10;                                                  \
    if (_result == limit) {                                                    \
      std::cout << "maximum number of errors exceeded" << std::endl;           \
      throw std::runtime_error("maximum number of errors exceeded");           \
    }                                                                          \
  } while (0)

#define TEST_END return _result
#define TEST_FUNC(F)                                                           \
  do {                                                                         \
    auto _result_temp = F;                                                     \
    if (_result_temp) {                                                        \
      std::cout.precision(std::numeric_limits<double>::max_digits10);          \
      std::cout << __FILE__ << ":" << __LINE__ << " Failure" << std::endl;     \
      std::cout << #F << " failed" << std::endl;                               \
    }                                                                          \
    if (_result_temp) {                                                        \
      INCREASE;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_NEAR_ABS(VAL1, VAL2, TOL)                                       \
  do {                                                                         \
    auto [is_near, tol] = detail::expect_near_abs(VAL1, VAL2, TOL);            \
    if (!is_near) {                                                            \
      std::cout.precision(std::numeric_limits<double>::max_digits10);          \
      std::cout << __FILE__ << ":" << __LINE__ << " Failure" << std::endl;     \
      std::cout << "Absolute difference " << tol << " exceeds " << TOL         \
                << ", where" << std::endl;                                     \
      std::cout << "val1 evaluates to " << VAL1 << std::endl;                  \
      std::cout << "val2 evaluates to " << VAL2 << std::endl;                  \
      INCREASE;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_NEAR_REL(VAL1, VAL2, TOL)                                       \
  do {                                                                         \
    auto [is_near, tol] = detail::expect_near_rel(VAL1, VAL2, TOL);            \
    if (!is_near) {                                                            \
      std::cout.precision(std::numeric_limits<double>::max_digits10);          \
      std::cout << __FILE__ << ":" << __LINE__ << " Failure" << std::endl;     \
      std::cout << "Relative difference " << tol << " exceeds " << TOL         \
                << ", where" << std::endl;                                     \
      std::cout << "val1 evaluates to " << VAL1 << std::endl;                  \
      std::cout << "val2 evaluates to " << VAL2 << std::endl;                  \
      INCREASE;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_NEAR_REL_ARRAY(VAL1, VAL2, TOL)                                 \
  do {                                                                         \
    constexpr std::size_t size = std::min(VAL1.size(), VAL2.size());           \
    for (std::size_t i = 0; i < size; i++) {                                   \
      EXPECT_NEAR_REL(VAL1[i], VAL2[i], TOL);                                  \
    }                                                                          \
  } while (0)

#define EXPECT_LESS_THAN(VAL1, VAL2)                                           \
  do {                                                                         \
    if (!(VAL1 < VAL2)) {                                                      \
      std::cout.precision(std::numeric_limits<double>::max_digits10);          \
      std::cout << __FILE__ << ":" << __LINE__ << " Failure" << std::endl;     \
      std::cout << "Expected " << VAL1 << " < " << VAL2 << std::endl;          \
      INCREASE;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_EQUAL(VAL1, VAL2)                                               \
  do {                                                                         \
    if ((VAL1 != VAL2)) {                                                      \
      std::cout.precision(std::numeric_limits<double>::max_digits10);          \
      std::cout << __FILE__ << ":" << __LINE__ << " Failure" << std::endl;     \
      std::cout << "Expected " << #VAL1 << " == " << #VAL2 << std::endl;       \
      INCREASE;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_TRUE(VAL) EXPECT_EQUAL(VAL, true)
#define EXPECT_FALSE(VAL) EXPECT_EQUAL(VAL, false)

#define EXPECT_NOT_EQUAL(VAL1, VAL2)                                           \
  do {                                                                         \
    if ((VAL1 == VAL2)) {                                                      \
      std::cout.precision(std::numeric_limits<double>::max_digits10);          \
      std::cout << __FILE__ << ":" << __LINE__ << " Failure" << std::endl;     \
      std::cout << "Expected " << #VAL1 << " != " << #VAL2 << std::endl;       \
      INCREASE;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_EQUAL_ARRAY(VAL1, VAL2)                                         \
  do {                                                                         \
    constexpr std::size_t size = std::min(VAL1.size(), VAL2.size());           \
    for (std::size_t i = 0; i < size; i++) {                                   \
      EXPECT_EQUAL(VAL1[i], VAL2[i]);                                          \
    }                                                                          \
  } while (0)

#endif // TEST_TOOLS_HPP
