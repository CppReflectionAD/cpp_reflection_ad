// A name that isn't a parameter or local is folded to a constant, even when a
// parameter or local of the same body has the same identifier.
//
// The engine looks names up by identifier, because a parameter's reflection
// never compares equal to the one its uses give. Before the lookup was limited
// to declarations whose parent is a function, on clang `ns::k * k` found the
// local `k` twice and computed x^4, with a derivative of 108 at x = 3, not 12,
// and no error. GCC reflects a read of a constexpr variable as its value, so it
// never looked `k` up.
//
// The lookup also keeps two kinds of declaration away from identifier_of, which
// rejects both. The parent check leaves out a variable template specialization,
// on clang, and has_identifier skips an unnamed parameter, on both compilers.
#include <test_simple_include.hpp>

#include <ad/forward_derivative.h>
#include <ad/reverse_derivative.h>
#include <numbers>

namespace ns {
constexpr double k = 2.0;
}
constexpr double k = 2.0;
struct Cfg {
  static constexpr double k = 2.0;
};
enum class Scale { two = 2 };
enum Plain { three = 3 };
template <typename T> constexpr T scale = T(2);

inline double namespace_member(double x) {
  double k = x * x;
  return ns::k * k;
}

inline double global(double x) {
  double k = x * x;
  return ::k * k;
}

inline double static_member(double x) {
  double k = x * x;
  return Cfg::k * k;
}

inline double library_constant(double x) {
  double pi = x;
  return std::numbers::pi * pi;
}

inline double scoped_enumerator(double x) {
  double two = x * x;
  return static_cast<double>(Scale::two) * two;
}

inline double plain_enumerator(double x) {
  double three = x * x;
  return static_cast<double>(Plain::three) * three;
}

inline double variable_template(double x) { return scale<double> * x; }

template <double K> inline double template_parameter(double x) { return K * x; }

inline double unnamed_parameter(double x, double) {
  double y = 2.0 * x;
  return y;
}

inline double unnamed_in_helper(double x, double) { return k * x; }
inline double calls_unnamed(double x) { return unnamed_in_helper(x, 1.0); }

// One helper deep, with the caller's parameter and local both named `k`.
inline double reads_global(double x) { return ::k * x; }
inline double shadowing_caller(double k) {
  double r = reads_global(k);
  return r + k;
}

int main() {
  // 2x^2: d/dx = 4x
  EXPECT_NEAR_ABS((ad::forward_derivative<^^namespace_member, 0>(3.0)), 12.0,
                  1e-12);
  EXPECT_NEAR_ABS((ad::gradient_reverse<^^namespace_member>(3.0))[0], 12.0,
                  1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^global, 0>(3.0)), 12.0, 1e-12);
  EXPECT_NEAR_ABS((ad::gradient_reverse<^^global>(3.0))[0], 12.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^static_member, 0>(3.0)), 12.0,
                  1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^scoped_enumerator, 0>(3.0)), 12.0,
                  1e-12);

  // 3x^2: d/dx = 6x
  EXPECT_NEAR_ABS((ad::forward_derivative<^^plain_enumerator, 0>(3.0)), 18.0,
                  1e-12);

  // pi x: d/dx = pi
  EXPECT_NEAR_ABS((ad::forward_derivative<^^library_constant, 0>(3.0)),
                  std::numbers::pi, 1e-12);

  // 2x: d/dx = 2
  EXPECT_NEAR_ABS((ad::forward_derivative<^^variable_template, 0>(3.0)), 2.0,
                  1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^template_parameter<2.0>, 0>(3.0)),
                  2.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^unnamed_parameter, 0>(3.0, 1.0)),
                  2.0, 1e-12);
  EXPECT_NEAR_ABS((ad::forward_derivative<^^calls_unnamed, 0>(3.0)), 2.0,
                  1e-12);

  // 2k + k: d/dk = 3
  EXPECT_NEAR_ABS((ad::forward_derivative<^^shadowing_caller, 0>(3.0)), 3.0,
                  1e-12);

  TEST_END;
}
