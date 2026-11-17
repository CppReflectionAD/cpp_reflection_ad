// TEST-PASSES-ON: clang
//
// Known GCC gap: AD through a fold expression over a function parameter pack.
//
// The fold itself is not the problem: instantiating fold_sum_sq<double,
// double, double> expands it into ordinary operators, (x0*x0) + ((x1*x1) +
// (x2*x2)). But the expanded parameters have no name, and the AD resolves a
// use of a parameter by binding the parameter's variable
// (ad::meta::parameter_variable) and matching by identity. The clang fork's
// variable_of works outside the function's body, so that resolves each pack
// element to its own slot. GCC's follows P3096 and only works inside the body,
// so on GCC the AD falls back on the name, which can't tell the elements
// apart.
//
// On GCC, tests/CMakeLists.txt expects this file to fail with the errors in
// fold_expression_ad.fail.txt. The TEST-PASSES-ON line makes it an ordinary
// test on clang, which passes every check. The checks are static_asserts so
// that GCC fails at compile time however the gap shows (no resolution, or
// every use resolving to one element). When GCC's variable_of works outside
// the body too, delete the GCC branch of parameter_variable and the name
// fallback in autograd.h, move this file to tests/common/, drop that line and
// delete the .fail.txt.
#include <ad/forward_derivative.h>
#include <ad/recursive_higher_order_derivative.h>
#include <ad/reverse_derivative.h>

// Sum of squares, so that each partial derivative (2 * x_i) is different.
template <typename... Ts> constexpr double fold_sum_sq(Ts... xs) {
  return ((xs * xs) + ...);
}

constexpr auto f = ^^fold_sum_sq<double, double, double>;
constexpr double x = 0.5;
constexpr double y = 2.0;
constexpr double z = -1.5;

static_assert(ad::forward_derivative<f, 0>(x, y, z) == 2 * x);
static_assert(ad::forward_derivative<f, 1>(x, y, z) == 2 * y);
static_assert(ad::forward_derivative<f, 2>(x, y, z) == 2 * z);

static_assert(ad::gradient_reverse<f>(x, y, z)[0] == 2 * x);
static_assert(ad::gradient_reverse<f>(x, y, z)[1] == 2 * y);
static_assert(ad::gradient_reverse<f>(x, y, z)[2] == 2 * z);

static_assert(ad::partial_derivative<f, 1, 1>(x, y, z) == 2.0);
static_assert(ad::partial_derivative<f, 0, 2>(x, y, z) == 0.0);

int main() {}
