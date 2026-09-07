#include "mc_sim/normal_distribution.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
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
  WorstCase worst_x;

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

    update_worst(worst_x, x, x, roundtrip_x);
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

    update_worst(worst_x, x, x, roundtrip_x);
    if (std::abs(roundtrip_x - x) > kXtolWide) {
      ++x_wide_failures;
    }
  }

  std::cout << std::setprecision(17);
  std::cout << "CDF(CDF_inverse(u)) quick check\n";
  std::cout << "  tolerance: " << kUtol << "\n";
  std::cout << "  max abs error: " << worst_u.abs_error
            << " at u=" << worst_u.input << " (got " << worst_u.actual << ")\n";
  std::cout << "  failures: " << u_failures << "\n\n";

  std::cout << "CDF_inverse(CDF(x)) quick check\n";
  std::cout << "  central tolerance [-6,6]: " << kXtolCentral << "\n";
  std::cout << "  wide tolerance [-8,8]: " << kXtolWide << "\n";
  std::cout << "  max abs error: " << worst_x.abs_error
            << " at x=" << worst_x.input << " (got " << worst_x.actual << ")\n";
  std::cout << "  central failures: " << x_central_failures << "\n";
  std::cout << "  wide failures: " << x_wide_failures << "\n";

  if (u_failures == 0 && x_central_failures == 0 && x_wide_failures == 0) {
    std::cout << "\nPASS\n";
    return 0;
  }

  std::cout << "\nFAIL\n";
  return 1;
}
