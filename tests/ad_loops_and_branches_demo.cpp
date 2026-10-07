// ad_loops_and_branches_demo.cpp — AD through `if`/`else` and compile-time-
// bounded `for` loops (statements, as opposed to the conditional operator
// already covered by ad_control_flow_demo.cpp).
//
// `if`/`else` reuses the ternary's guard/Select machinery at statement
// granularity (see lower_if in autograd.h): each branch is lowered under a
// narrowed guard, and any outer variable reassigned in either branch gets a
// Select at the join. A `for` loop is lowered by unrolling: its trip count
// must be a compile-time-provable constant, computed by directly simulating
// the loop header at reflection time.

#include "test_simple_include.hpp"

#include "autograd.h"

#include "functions/7-loops.h"
#include "functions/8-branches.h"

#include <cmath>

namespace {

// Central difference, for checking derivatives independent of the AD engine.
template <class F> double fd(F f, double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

template <std::meta::info Fn> consteval bool lowers_to(ad::OpKind op) {
  for (const ad::Node &n : ad::build_nodes<Fn>())
    if (n.op == op)
      return true;
  return false;
}

}  // namespace

// `if`/`else` desugars entirely into the existing Select -- no new OpKind.
static_assert(lowers_to<^^abs_via_if>(ad::OpKind::Select));
// A `for` loop is unrolled, so its body's ops appear directly, repeated --
// `repeated_add`'s body has one Add per iteration, times 5.
static_assert(lowers_to<^^repeated_add>(ad::OpKind::Add));

int main() {
  // --- if/else: both branches reassign the same variable -------------------
  {
    EXPECT_NEAR_ABS((ad::forward_derivative<^^abs_via_if, 0>(2.0)), 1.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^abs_via_if, 0>(-2.0)), -1.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^abs_via_if>(2.0)[0], 1.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^abs_via_if>(-2.0)[0], -1.0, 1e-12);
  }

  // --- if with no else: the untouched branch keeps the pre-if value --------
  {
    EXPECT_NEAR_ABS((ad::forward_derivative<^^clamp_floor_via_if, 0>(5.0, 1.0)),
                    1.0, 1e-12);
    EXPECT_NEAR_ABS((ad::forward_derivative<^^clamp_floor_via_if, 0>(0.0, 1.0)),
                    0.0, 1e-12);
  }

  // --- init-statement + else-if chaining ------------------------------------
  for (double x : {0.5, 1.5, 3.0}) {
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^staircase_via_if>(x)[0],
                    fd(staircase_via_if, x), 1e-6);
  }

  // --- for loops: unrolled, so the derivative is exact ----------------------
  {
    EXPECT_NEAR_ABS((ad::forward_derivative<^^repeated_add, 0>(3.0)), 5.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^repeated_add>(3.0)[0], 5.0, 1e-12);

    EXPECT_NEAR_ABS((ad::forward_derivative<^^countdown_sum, 0>(3.0)), 1.0, 1e-12);
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^countdown_sum>(3.0)[0], 1.0, 1e-12);

    EXPECT_NEAR_ABS(ad::gradient_reverse<^^nested_sum>(3.0)[0], 9.0, 1e-12);
  }

  // --- a loop body that itself branches --------------------------------------
  for (double x : {-1.0, 0.5, 1.5, 2.5}) {
    EXPECT_NEAR_ABS(ad::gradient_reverse<^^sum_of_abs_steps>(x)[0],
                    fd(sum_of_abs_steps, x), 1e-6);
  }

  TEST_END;
}
