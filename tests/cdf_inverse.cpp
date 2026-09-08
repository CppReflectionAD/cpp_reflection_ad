#include "mc_sim/normal_distribution.hpp"
#include <test_simple_include.hpp>

#include <cmath>
#include <vector>

namespace {

struct WorstCase {
  double input = 0.0;
  double expected = 0.0;
  double actual = 0.0;
  double abs_error = 0.0;
};

void update_worst(WorstCase &worst, double input, double expected,
                  double actual) {
  const double err = std::abs(actual - expected);
  if (err > worst.abs_error) {
    worst = WorstCase{input, expected, actual, err};
  }
}

} // namespace

int main() {
  constexpr double kUtol = 5e-10;
  constexpr double kXtolCentral = 3e-7;
  constexpr double kXtolWide = 2e-2;

  WorstCase worst_u;
  WorstCase worst_x_central;
  WorstCase worst_x_wide;

  int u_failures = 0;
  int x_central_failures = 0;
  int x_wide_failures = 0;

  // Check u -> inv -> cdf on a dense grid in (0, 1).
  constexpr int kUSteps = 200000;
  const double u_min = 1e-12;
  const double u_max = 1.0 - 1e-12;
  for (int i = 0; i <= kUSteps; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(kUSteps);
    const double u = u_min + (u_max - u_min) * t;
    const double roundtrip_u = mcsim::CDF(mcsim::CDF_inverse(u));

    update_worst(worst_u, u, u, roundtrip_u);
    if (std::abs(roundtrip_u - u) > kUtol) {
      ++u_failures;
    }
  }

  // Include a few hand-picked tail points.
  const std::vector<double> tail_u = {
      1e-15,      1e-14,       1e-10,       1e-6,
      1.0 - 1e-6, 1.0 - 1e-10, 1.0 - 1e-14, 1.0 - 1e-15,
  };

  for (double u : tail_u) {
    const double roundtrip_u = mcsim::CDF(mcsim::CDF_inverse(u));
    update_worst(worst_u, u, u, roundtrip_u);
    if (std::abs(roundtrip_u - u) > kUtol) {
      ++u_failures;
    }
  }

  // Strict check in the central range where round-trip is expected to be tight.
  constexpr int kXCentralSteps = 200000;
  constexpr double x_central_min = -6.0;
  constexpr double x_central_max = 6.0;
  for (int i = 0; i <= kXCentralSteps; ++i) {
    const double t =
        static_cast<double>(i) / static_cast<double>(kXCentralSteps);
    const double x = x_central_min + (x_central_max - x_central_min) * t;
    const double roundtrip_x = mcsim::CDF_inverse(mcsim::CDF(x));

    update_worst(worst_x_central, x, x, roundtrip_x);
    if (std::abs(roundtrip_x - x) > kXtolCentral) {
      ++x_central_failures;
    }
  }

  // Wide-range sanity check: tails can amplify floating-point effects.
  constexpr int kXSteps = 200000;
  constexpr double x_min = -8.0;
  constexpr double x_max = 8.0;
  for (int i = 0; i <= kXSteps; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(kXSteps);
    const double x = x_min + (x_max - x_min) * t;
    const double roundtrip_x = mcsim::CDF_inverse(mcsim::CDF(x));

    update_worst(worst_x_wide, x, x, roundtrip_x);
    if (std::abs(roundtrip_x - x) > kXtolWide) {
      ++x_wide_failures;
    }
  }

  EXPECT_TRUE(worst_u.abs_error <= kUtol);
  EXPECT_EQUAL(u_failures, 0);

  EXPECT_TRUE(worst_x_central.abs_error <= kXtolCentral);
  EXPECT_EQUAL(x_central_failures, 0);

  EXPECT_TRUE(worst_x_wide.abs_error <= kXtolWide);
  EXPECT_EQUAL(x_wide_failures, 0);

  TEST_END;
}
