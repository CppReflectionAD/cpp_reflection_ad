// Compile-time verification of the Runge function f(x) = 1 / (x*x + 1).
//
// Two properties, each proved by the interval engine (ad/is_continuous.hpp)
// while compiling; if either fails, this file doesn't compile:
//
//   safe:    the division never sees a zero denominator, so f is defined and
//            continuous on the whole box (is_continuous_on);
//   bounded: every value f takes on the box lies in [0, 1] (range_on).
//
// The engine propagates an interval through each node of the reflected body,
// so a `static_assert` on its answer covers every x in the box, not a sample.
// It over-approximates, so a "safe" or a bound it proves holds; it can miss
// one that is true. Its arithmetic rounds to nearest rather than outward, so
// an end of an enclosure can be off by an ulp: the bounds proved below leave
// far more room than that.

#include <ad/is_continuous.hpp>
#include <test_simple_include.hpp>

inline double runge(double x) { return 1.0 / (x * x + 1.0); }

// For contrast: a pole at x = ±1.
inline double runge_pole(double x) { return 1.0 / (x * x - 1.0); }

// --- safe ------------------------------------------------------------------
// x*x is a square, so x*x + 1 >= 1 and the denominator is never 0. The engine
// sees that x*x multiplies a node by itself and gives it [0, 100] on
// [-10, 10]; as two independent factors it would be [-100, 100], and the
// denominator [-99, 101] would contain 0.
static_assert(ad::is_continuous_on<^^runge>(ad::Interval{-10.0, 10.0}));
static_assert(ad::is_continuous_on<^^runge>(ad::Interval{0.0, 0.0}));
// As wide as doubles allow without x*x overflowing to infinity.
static_assert(ad::is_continuous_on<^^runge>(ad::Interval{-1e150, 1e150}));

// The checker isn't vacuous: it rejects the pole wherever the box reaches it.
static_assert(!ad::is_continuous_on<^^runge_pole>(ad::Interval{-2.0, 2.0}));
static_assert(!ad::is_continuous_on<^^runge_pole>(ad::Interval{0.5, 1.0}));
static_assert(ad::is_continuous_on<^^runge_pole>(ad::Interval{1.5, 2.0}));

// --- bounded ---------------------------------------------------------------
// 0 < f(x) <= 1 for every x in the box.
constexpr ad::Interval runge_range =
    ad::range_on<^^runge>(ad::Interval{-10.0, 10.0});
static_assert(runge_range.lo > 0.0);
static_assert(runge_range.hi <= 1.0);
// And the enclosure is tight here: f(±10) = 1/101 and f(0) = 1.
static_assert(runge_range.lo == 1.0 / 101.0);
static_assert(runge_range.hi == 1.0);

// Still in [0, 1] on the widest box.
constexpr ad::Interval runge_wide =
    ad::range_on<^^runge>(ad::Interval{-1e150, 1e150});
static_assert(runge_wide.lo >= 0.0 && runge_wide.hi <= 1.0);

// Away from 0 the bound tightens: on [2, 3], f is in [1/10, 1/5].
constexpr ad::Interval runge_tail =
    ad::range_on<^^runge>(ad::Interval{2.0, 3.0});
static_assert(runge_tail.lo == 1.0 / 10.0 && runge_tail.hi == 1.0 / 5.0);

int main() {
  // The detailed result agrees with the static_asserts.
  {
    constexpr auto r =
        ad::continuity_result<^^runge>(ad::Interval{-10.0, 10.0});
    EXPECT_TRUE(r.continuous);
    EXPECT_EQUAL(r.failing_node, -1);
  }

  // The pole is reported at the division.
  {
    constexpr auto r =
        ad::continuity_result<^^runge_pole>(ad::Interval{-2.0, 2.0});
    EXPECT_FALSE(r.continuous);
    EXPECT_EQUAL(r.failing_op, ad::OpKind::Div);
  }

  // Sampling the box at run time stays inside the proved enclosure.
  for (int i = 0; i <= 2000; ++i) {
    const double x = -10.0 + i * 0.01;
    const double y = runge(x);
    EXPECT_TRUE(runge_range.lo <= y && y <= runge_range.hi);
  }

  TEST_END;
}
