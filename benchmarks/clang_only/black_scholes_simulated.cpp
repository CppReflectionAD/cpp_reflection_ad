#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../../tests/clang_only/discontinuity_analysis.hpp"
#include "../../tests/clang_only/is_continuous.hpp"
#include "../../tests/forward_derivative.h"
#include "../../tests/is_invertible.hpp"
#include "../../tests/mc_sim/black_scholes.hpp"
#include "../../tests/mc_sim/evolve_black_scholes.hpp"

double g_smoothed_dirac_width = 1.0;

double smooth_dirac(double x, double width) {
  if (width <= 0.0) {
    return 0.0;
  }

  constexpr double inv_sqrt_pi =
      0.564189583547756286948079451560772585844050629329;
  const double scaled = x / width;
  return inv_sqrt_pi * std::exp(-(scaled * scaled)) / width;
}

// first case: digital call option
template <double Strike = 100.0> double digital_call_payoff(double spot) {
  return (spot > Strike) ? 1.0 : 0.0;
}

template <double Strike = 100.0>
double digital_call_payoff_derivative(double spot) {
  return smooth_dirac(spot - Strike, g_smoothed_dirac_width);
}

template <double Strike = 100.0>
double digital_call_closed_form(double spot0, double rate, double vol,
                                double maturity_years) {
  // black_scholes.hpp formulas are expressed on forward variables.
  const double forward = spot0 * std::exp(rate * maturity_years);
  const double discount = std::exp(-rate * maturity_years);
  return discount * digital_call_price(forward, Strike, vol, maturity_years);
}

template <double Strike = 100.0> double digital_put_payoff(double spot) {
  return (spot < Strike) ? 1.0 : 0.0;
}

template <double Strike = 100.0>
double digital_put_payoff_derivative(double spot) {
  return -smooth_dirac(spot - Strike, g_smoothed_dirac_width);
}

template <double Strike = 100.0>
double digital_put_closed_form(double spot0, double rate, double vol,
                               double maturity_years) {
  const double forward = spot0 * std::exp(rate * maturity_years);
  const double discount = std::exp(-rate * maturity_years);
  return discount * digital_put_price(forward, Strike, vol, maturity_years);
}

// second case: digital AND call
template <double Strike = 100.0> double digital_and_call_payoff(double spot) {
  return (spot > Strike) ? (1.0 + spot - Strike) : 0.0;
}

template <double Strike = 100.0>
double digital_and_call_payoff_derivative(double spot) {
  return ((spot > Strike) ? 1.0 : 0.0) +
         smooth_dirac(spot - Strike, g_smoothed_dirac_width);
}

template <double Strike = 100.0>
double digital_and_call_closed_form(double spot0, double rate, double vol,
                                    double maturity_years) {
  const double forward = spot0 * std::exp(rate * maturity_years);
  const double discount = std::exp(-rate * maturity_years);
  return discount * (digital_call_price(forward, Strike, vol, maturity_years) +
                     call_price(forward, Strike, vol, maturity_years));
}

template <double Strike1 = 99.0, double Strike2 = 101.0>
double double_digital_payoff(double spot) {
  return ((spot > Strike1) ? 1.0 : 0.0) - ((spot > Strike2) ? 1.0 : 0.0);
}

template <double Strike1 = 99.0, double Strike2 = 101.0>
double double_digital_payoff_derivative(double spot) {
  return smooth_dirac(spot - Strike1, g_smoothed_dirac_width) -
         smooth_dirac(spot - Strike2, g_smoothed_dirac_width);
}

template <double Strike1 = 99.0, double Strike2 = 101.0>
double double_digital_closed_form(double spot0, double rate, double vol,
                                  double maturity_years) {

  const double forward = spot0 * std::exp(rate * maturity_years);
  const double discount = std::exp(-rate * maturity_years);
  return discount * (digital_call_price(forward, Strike1, vol, maturity_years) -
                     digital_call_price(forward, Strike2, vol, maturity_years));
}

template <double Strike1 = 99.0, double Strike2 = 101.0>
double double_digital_butterfly_payoff(double spot) {
  return ((spot > Strike1) ? -1.0 : (Strike1 - spot)) +
         ((spot > Strike2) ? 1.0 + (spot - Strike2) : 0.0);
}

template <double Strike1 = 99.0, double Strike2 = 101.0>
double double_digital_butterfly_payoff_derivative(double spot) {
  return ((spot < Strike1) ? -1.0 : 0.0) -
         smooth_dirac(spot - Strike1, g_smoothed_dirac_width) +
         smooth_dirac(spot - Strike2, g_smoothed_dirac_width) +
         ((spot > Strike2) ? 1.0 : 0.0);
}

template <double Strike1 = 99.0, double Strike2 = 101.0>
double double_digital_butterfly_closed_form(double spot0, double rate,
                                            double vol, double maturity_years) {

  const double forward = spot0 * std::exp(rate * maturity_years);
  const double discount = std::exp(-rate * maturity_years);
  return discount *
         (put_price(forward, Strike1, vol, maturity_years) +
          -digital_call_price(forward, Strike1, vol, maturity_years) +
          digital_call_price(forward, Strike2, vol, maturity_years) +
          call_price(forward, Strike2, vol, maturity_years));
}

template <double Strike = 100.0>
double nonlinear_digital_call_payoff(double spot) {
  return (spot > Strike) ? ((spot + 3.0) / (spot + 2.0)) : 0.0;
}

template <double Strike = 100.0>
double nonlinear_digital_call_payoff_derivative(double spot) {
  const double jump = (Strike + 3.0) / (Strike + 2.0);
  const double smooth_jump =
      smooth_dirac(spot - Strike, g_smoothed_dirac_width);
  const double regular_part =
      (spot > Strike) ? (-1.0 / ((spot + 2.0) * (spot + 2.0))) : 0.0;
  return regular_part + jump * smooth_jump;
}

using TimePoint = std::chrono::system_clock::time_point;
using Days = std::chrono::duration<std::int64_t, std::ratio<86400>>;

double year_fraction_act365(const TimePoint &from, const TimePoint &to) {
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(to - from)
          .count();
  return seconds / (365.0 * 24.0 * 3600.0);
}

std::vector<TimePoint> build_date_grid(const TimePoint &start,
                                       const TimePoint &end,
                                       Days step = Days{30}) {
  std::vector<TimePoint> dates;
  dates.push_back(start);
  for (auto d = start + step; d < end; d += step) {
    dates.push_back(d);
  }

  if (dates.back() != end) {
    dates.push_back(end);
  }

  return dates;
}

inline double g_last_step(double spot0, double factors_except_last, double r,
                          double vol, double dt_last, double normal_z) {
  return spot0 * factors_except_last *
         evolve_black_scholes_normal(r, vol, dt_last, normal_z);
}

template <std::meta::info FinalPayoffFn, std::meta::info FinalPayoffDeltaFn>
std::array<double, 2> monte_carlo_engine_smoothed_dirac(
    double spot0, double r, double vol, double maturity, std::size_t num_paths,
    std::size_t sim_per_path, std::uint32_t seed) {
  if (spot0 <= 0.0 || vol <= 0.0 || maturity <= 0.0 || num_paths == 0 ||
      sim_per_path == 0) {
    return {0.0, 0.0};
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  const double discount = std::exp(-r * maturity);
  const double dt_regular = maturity / static_cast<double>(sim_per_path);
  const double dt_stub =
      maturity - dt_regular * static_cast<double>(sim_per_path - 1);
  std::vector<double> dts(sim_per_path, dt_regular);
  dts.back() = dt_stub;

  double payoff_sum = 0.0;
  double payoff_delta_sum = 0.0;
  for (std::size_t path = 0; path < num_paths; ++path) {
    std::vector<double> normals_prefix(sim_per_path, 0.0);

    for (double &z : normals_prefix) {
      z = mcsim::CDF_inverse(unif(rng));
    }

    double spot_before_last = spot0;
    double factors_except_last = 1.0;
    for (std::size_t step = 0; step + 1 < sim_per_path; ++step) {
      const double factor =
          evolve_black_scholes_normal(r, vol, dts[step], normals_prefix[step]);
      spot_before_last *= factor;
      factors_except_last *= factor;
    }

    double spot = g_last_step(spot0, factors_except_last, r, vol, dts.back(),
                              normals_prefix.back());
    payoff_sum += static_cast<double>([:FinalPayoffFn:](spot));

    double const spot_d = ad::forward_derivative<^^g_last_step, 0>(
        spot0, factors_except_last, r, vol, dts.back(), normals_prefix.back());
    payoff_delta_sum +=
        spot_d * static_cast<double>([:FinalPayoffDeltaFn:](spot));
  }

  return {discount * (payoff_sum / static_cast<double>(num_paths)),
          discount * (payoff_delta_sum / static_cast<double>(num_paths))};
}

template <std::meta::info FinalPayoffFn>
std::array<double, 3> monte_carlo_engine(double spot0, double r, double vol,
                                         double maturity, std::size_t num_paths,
                                         std::size_t sim_per_path,
                                         std::uint32_t seed) {
  if (spot0 <= 0.0 || vol <= 0.0 || maturity <= 0.0 || num_paths == 0 ||
      sim_per_path == 0) {
    return {0.0, 0.0};
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  const double discount = std::exp(-r * maturity);
  const double dt_regular = maturity / static_cast<double>(sim_per_path);
  const double dt_stub =
      maturity - dt_regular * static_cast<double>(sim_per_path - 1);
  std::vector<double> dts(sim_per_path, dt_regular);
  dts.back() = dt_stub;

  double payoff_sum = 0.0;
  double payoff_delta_sum = 0.0;
  double correction_sum = 0.0;

  for (std::size_t path = 0; path < num_paths; ++path) {
    std::vector<double> normals_prefix(sim_per_path, 0.0);

    for (double &z : normals_prefix) {
      z = mcsim::CDF_inverse(unif(rng));
    }

    double factors_except_last = 1.0;
    for (std::size_t step = 0; step + 1 < sim_per_path; ++step) {
      const double factor =
          evolve_black_scholes_normal(r, vol, dts[step], normals_prefix[step]);
      factors_except_last *= factor;
    }

    double spot = g_last_step(spot0, factors_except_last, r, vol, dts.back(),
                              normals_prefix.back());
    payoff_sum += [:FinalPayoffFn:](spot);

    // we cannot yet reflect the composed payoff function, so we apply chain
    // rule manually here
    payoff_delta_sum += ad::forward_derivative<^^g_last_step, 0>(
                            spot0, factors_except_last, r, vol, dts.back(),
                            normals_prefix.back()) *
                        ad::forward_derivative<FinalPayoffFn, 0>(spot);

    // Use runtime version to accept dynamic parameters
    constexpr auto discontinuities =
        ad::get_discontinuity_points_and_amplitudes<FinalPayoffFn, 0>();

    for (std::size_t i = 0; i < discontinuities.size(); ++i) {
      // Tweak the last draw so terminal spot lands exactly on strike, using
      // the generic inverse machinery rather than an explicit closed form.
      const double z_star = ad::inverse_of_wrt<^^g_last_step, 5>(
          discontinuities.point(i), spot0, factors_except_last, r, vol,
          dts.back());
      const double normal_pdf = mcsim::PDF(z_star);

      // Sifting term written through the normal draw z:
      // contribution = f_Z(z*) * (dg/dS0)/|dg/dz| with f_Z = phi.
      // Since u = Phi(z), this is equivalent to using |dg/du| in U-space.
      const double dg_d_spot0 = discontinuities.point(i) / spot0;
      const double dg_d_z = ad::forward_derivative<^^g_last_step, 5>(
          spot0, factors_except_last, r, vol, dts.back(), z_star);
      const double inv_abs_dg_d_u = normal_pdf / std::abs(dg_d_z);

      correction_sum +=
          dg_d_spot0 * inv_abs_dg_d_u * discontinuities.amplitude(i);
    }
  }

  return {discount * (payoff_sum / static_cast<double>(num_paths)),
          discount * (payoff_delta_sum / static_cast<double>(num_paths)),
          discount * (correction_sum / static_cast<double>(num_paths))};
}

int main(int argc, char **argv) {
  const TimePoint start{};
  const TimePoint maturity = start + Days{367};

  const double spot0 = 100.0;
  double strike = 100.0;
  const double rate = 0.03;
  const double vol = 0.20;
  std::size_t num_paths = 200000;
  std::size_t sim_per_path = 1;
  double tolerance = 1e-2;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value after " << flag << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " [--width W|-w W] [--paths N] [--sim-per-path N]"
                  << " [--tolerance X]" << std::endl;
        std::exit(1);
      }
      return argv[++i];
    };

    if (arg == "--width" || arg == "-w") {
      g_smoothed_dirac_width = std::stod(require_value(arg.c_str()));
    } else if (arg == "--paths") {
      num_paths = std::stoul(require_value(arg.c_str()));
    } else if (arg == "--sim-per-path") {
      sim_per_path = std::stoul(require_value(arg.c_str()));
    } else if (arg == "--tolerance") {
      tolerance = std::stod(require_value(arg.c_str()));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--width W|-w W] [--paths N] [--sim-per-path N]"
                << " [--tolerance X]" << std::endl;
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::cerr << "Usage: " << argv[0]
                << " [--width W|-w W] [--paths N] [--sim-per-path N]"
                << " [--tolerance X]" << std::endl;
      return 1;
    }
  }

  if (num_paths == 0) {
    std::cerr << "--paths must be a positive integer" << std::endl;
    return 1;
  }
  if (sim_per_path == 0) {
    std::cerr << "--sim-per-path must be a positive integer" << std::endl;
    return 1;
  }
  if (tolerance <= 0.0) {
    std::cerr << "--tolerance must be a positive number" << std::endl;
    return 1;
  }
  if (g_smoothed_dirac_width <= 0.0) {
    std::cerr << "--width must be a positive number" << std::endl;
    return 1;
  }

  const std::uint32_t seed = 42;
  const double maturity_years = year_fraction_act365(start, maturity);

  std::cout << "Smoothed Dirac width         : " << g_smoothed_dirac_width
            << "\n\n";

  // Templated function to run payoff test with closed-form and payoff functions
  auto run_payoff_test = [&]<std::meta::info ClosedFormFn,
                             std::meta::info PayoffFn,
                             std::meta::info PayoffDeltaFn>(const char *label) {
    const double mc_bump = 1e-2;
    const double cf_pv = [:ClosedFormFn:](spot0, rate, vol, maturity_years);
    const double cf_delta = ad::forward_derivative<ClosedFormFn, 0>(
        spot0, rate, vol, maturity_years);

    // Finite difference delta
    const double h = 1e-8;
    const double cf_pv_up = [:ClosedFormFn:](spot0 + h, rate, vol,
                                             maturity_years);
    const double cf_pv_down = [:ClosedFormFn:](spot0 - h, rate, vol,
                                               maturity_years);
    const double cf_delta_fd = (cf_pv_up - cf_pv_down) / (2.0 * h);

    const auto [mc_pv, mc_delta, mc_correction] = monte_carlo_engine<PayoffFn>(
        spot0, rate, vol, maturity_years, num_paths, sim_per_path, seed);
    const double mc_delta_corrected = mc_delta + mc_correction;

    const auto [mc_pv_smoothed, mc_delta_smoothed] =
        monte_carlo_engine_smoothed_dirac<PayoffFn, PayoffDeltaFn>(
            spot0, rate, vol, maturity_years, num_paths, sim_per_path, seed);

    const auto [mc_pv_up, _, __] =
        monte_carlo_engine<PayoffFn>(spot0 + mc_bump, rate, vol, maturity_years,
                                     num_paths, sim_per_path, seed);
    const auto [mc_pv_down, ___, ____] =
        monte_carlo_engine<PayoffFn>(spot0 - mc_bump, rate, vol, maturity_years,
                                     num_paths, sim_per_path, seed);
    const double mc_delta_bump = (mc_pv_up - mc_pv_down) / (2.0 * mc_bump);

    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "Closed-form (" << label << ") : " << cf_pv << "\n";
    std::cout << "Closed-form delta (analytic) : " << cf_delta << "\n";
    std::cout << "Closed-form delta (FD)       : " << cf_delta_fd << "\n";
    std::cout << "MC " << label << " price (legacy)        : " << mc_pv << "\n";
    std::cout << "MC Delta (raw pathwise)      : " << mc_delta << "\n";
    std::cout << "MC Correction term           : " << mc_correction << "\n";
    std::cout << "MC Delta (legacy corrected)  : " << mc_delta_corrected
              << "\n";
    std::cout << "MC " << label << " price (smoothed)      : " << mc_pv_smoothed
              << "\n";
    std::cout << "MC Delta (smoothed pathwise) : " << mc_delta_smoothed << "\n";
    std::cout << "MC Delta (bump/reprice)      : " << mc_delta_bump << "\n";
    std::cout << "Relative % PV error (legacy): "
              << (std::abs(mc_pv - cf_pv) / std::abs(cf_pv)) * 100 << "%\n";
    std::cout << "Relative % PV error (smooth): "
              << (std::abs(mc_pv_smoothed - cf_pv) / std::abs(cf_pv)) * 100
              << "%\n";
    std::cout << "Relative % Delta error (legacy): "
              << (std::abs(mc_delta_corrected - cf_delta) /
                  std::abs(cf_delta)) *
                     100
              << "%\n";
    std::cout << "Relative % Delta error (smooth): "
              << (std::abs(mc_delta_smoothed - cf_delta) / std::abs(cf_delta)) *
                     100
              << "%\n";
    std::cout << "Legacy corrected vs bump error %: "
              << (std::abs(mc_delta_corrected - mc_delta_bump) /
                  std::max(1e-16, std::abs(mc_delta_bump))) *
                     100
              << "%\n";
    std::cout << "Smoothed pathwise vs bump error %: "
              << (std::abs(mc_delta_smoothed - mc_delta_bump) /
                  std::max(1e-16, std::abs(mc_delta_bump))) *
                     100
              << "%\n";
    std::cout << "\n";
  };

  auto run_mc_only_payoff_test = [&]<std::meta::info PayoffFn,
                                     std::meta::info PayoffDeltaFn>(
                                     const char *label) {
    const double mc_bump = 1e-2;
    const auto [mc_pv, mc_delta, mc_correction] = monte_carlo_engine<PayoffFn>(
        spot0, rate, vol, maturity_years, num_paths, sim_per_path, seed);
    const double mc_delta_corrected = mc_delta + mc_correction;
    const auto [mc_pv_smoothed, mc_delta_smoothed] =
        monte_carlo_engine_smoothed_dirac<PayoffFn, PayoffDeltaFn>(
            spot0, rate, vol, maturity_years, num_paths, sim_per_path, seed);
    const auto [mc_pv_up, _, __] =
        monte_carlo_engine<PayoffFn>(spot0 + mc_bump, rate, vol, maturity_years,
                                     num_paths, sim_per_path, seed);
    const auto [mc_pv_down, ___, ____] =
        monte_carlo_engine<PayoffFn>(spot0 - mc_bump, rate, vol, maturity_years,
                                     num_paths, sim_per_path, seed);
    const double mc_delta_bump = (mc_pv_up - mc_pv_down) / (2.0 * mc_bump);

    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "MC " << label << " price (legacy)        : " << mc_pv << "\n";
    std::cout << "MC Delta (raw pathwise)      : " << mc_delta << "\n";
    std::cout << "MC Correction term           : " << mc_correction << "\n";
    std::cout << "MC Delta (legacy corrected)  : " << mc_delta_corrected
              << "\n";
    std::cout << "MC " << label << " price (smoothed)      : " << mc_pv_smoothed
              << "\n";
    std::cout << "MC Delta (smoothed pathwise) : " << mc_delta_smoothed << "\n";
    std::cout << "MC Delta (bump/reprice)      : " << mc_delta_bump << "\n";
    std::cout << "Legacy corrected vs bump error %: "
              << (std::abs(mc_delta_corrected - mc_delta_bump) /
                  std::max(1e-16, std::abs(mc_delta_bump))) *
                     100
              << "%\n";
    std::cout << "Smoothed pathwise vs bump error %: "
              << (std::abs(mc_delta_smoothed - mc_delta_bump) /
                  std::max(1e-16, std::abs(mc_delta_bump))) *
                     100
              << "%\n";
    std::cout << "\n";
  };

  std::cout << "=== Digital Call ===\n\n";

  run_payoff_test.template
  operator()<^^digital_call_closed_form<100.0>, ^^digital_call_payoff<100.0>,
             ^^digital_call_payoff_derivative<100.0>>("digital call");

  std::cout << "=== Digital Put (negative slope in Heaviside) ===\n\n";

  run_payoff_test.template
  operator()<^^digital_put_closed_form<100.0>, ^^digital_put_payoff<100.0>,
             ^^digital_put_payoff_derivative<100.0>>("digital put");

  std::cout << "=== Digital Call And Call ===\n\n";

  run_payoff_test.template operator()<
      ^^digital_and_call_closed_form<100.0>, ^^digital_and_call_payoff<100.0>,
      ^^digital_and_call_payoff_derivative<100.0>>("digital and call");

  std::cout << "=== Double Digital ===\n\n";

  run_payoff_test.template
  operator()<^^double_digital_closed_form<99.0, 101.0>,
             ^^double_digital_payoff<99.0, 101.0>,
             ^^double_digital_payoff_derivative<99.0, 101.0>>("double digital");

  std::cout << "=== Double Digital Butterfly===\n\n";

  run_payoff_test.template
  operator()<^^double_digital_butterfly_closed_form<99.0, 101.0>,
             ^^double_digital_butterfly_payoff<99.0, 101.0>,
             ^^double_digital_butterfly_payoff_derivative<99.0, 101.0>>(
      "double digital butterfly");

  std::cout << "=== Nonlinear Digital Call (no closed form) ===\n\n";

  run_mc_only_payoff_test
      .template operator()<^^nonlinear_digital_call_payoff<100.0>,
                           ^^nonlinear_digital_call_payoff_derivative<100.0>>(
          "nonlinear digital call");

  std::cout << "=== Nonlinear Digital Call (no closed form) ===\n\n";

  run_mc_only_payoff_test.template
  operator()<^^nonlinear_digital_call_payoff<100.0>>("nonlinear digital call");

  bool show_convergence = false;
  if (show_convergence) {
    std::cout << "=== Convergence Analysis ===\n\n";

    const double cf_pv_spread1 = [:^^double_digital_closed_form<99.0, 101.0>:](
        spot0, rate, vol, maturity_years);
    const double cf_delta_spread1 =
        ad::forward_derivative<^^double_digital_closed_form<99.0, 101.0>, 0>(
            spot0, rate, vol, maturity_years);

    std::cout.precision(10);
    std::cout << "Closed-form PV:    " << cf_pv_spread1 << "\n";
    std::cout << "Closed-form delta: " << cf_delta_spread1 << "\n\n";

    std::cout.precision(6);
    std::cout << std::scientific
              << "Paths\t\tMC PV\t\t\tPV Error %\tMC Delta\t\tDelta Error%\n ";
    std::cout << "=====\t\t=====\t\t\t=========\t========\t\t=============\n";

    const std::vector<std::size_t> path_counts = {
        100,    500,     1000,    5000,    10000,   50000,    100000,
        500000, 1000000, 2000000, 4000000, 8000000, 10000000, 15000000};

    for (std::size_t paths : path_counts) {
      const auto [mc_pv, mc_delta] = monte_carlo_engine_smoothed_dirac<
          ^^double_digital_payoff<99.0, 101.0>,
          ^^double_digital_payoff_derivative<99.0, 101.0>>(
          spot0, rate, vol, maturity_years, paths, sim_per_path, seed);
      const double pv_error_pct =
          (std::abs(mc_pv - cf_pv_spread1) / cf_pv_spread1) * 100;
      const double delta_error_pct =
          (std::abs(mc_delta - cf_delta_spread1) / cf_delta_spread1) * 100;

      std::cout << paths << "\t\t" << mc_pv << "\t" << pv_error_pct << "%\t\t"
                << mc_delta << "\t" << delta_error_pct << "%\n";
    }
    std::cout << std::defaultfloat;
  }

  // Timing benchmarks
  std::cout << "\n=== Timing Benchmarks ===\n\n";

  auto run_timing_benchmark = [&]<std::meta::info PayoffFn>(
                                  const char *label,
                                  std::size_t benchmark_paths) {
    // Warm-up
    monte_carlo_engine<PayoffFn>(spot0, rate, vol, maturity_years,
                                 std::min(benchmark_paths / 10, 10000UL),
                                 sim_per_path, seed);

    // Full MC engine timing
    auto t_start = std::chrono::high_resolution_clock::now();
    const auto [mc_pv, mc_delta, mc_correction] = monte_carlo_engine<PayoffFn>(
        spot0, rate, vol, maturity_years, benchmark_paths, sim_per_path, seed);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_total =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
            .count();

    // Time just payoff computation
    t_start = std::chrono::high_resolution_clock::now();
    double payoff_sum = 0.0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const double dt_regular =
        maturity_years / static_cast<double>(sim_per_path);
    const double dt_stub =
        maturity_years - dt_regular * static_cast<double>(sim_per_path - 1);
    std::vector<double> dts(sim_per_path, dt_regular);
    dts.back() = dt_stub;
    for (std::size_t path = 0; path < benchmark_paths; ++path) {
      std::vector<double> normals_prefix(sim_per_path, 0.0);
      for (double &z : normals_prefix) {
        z = mcsim::CDF_inverse(unif(rng));
      }
      double factors_except_last = 1.0;
      for (std::size_t step = 0; step + 1 < sim_per_path; ++step) {
        const double factor = evolve_black_scholes_normal(rate, vol, dts[step],
                                                          normals_prefix[step]);
        factors_except_last *= factor;
      }
      double spot = g_last_step(spot0, factors_except_last, rate, vol,
                                dts.back(), normals_prefix.back());
      payoff_sum += [:PayoffFn:](spot);
    }
    t_end = std::chrono::high_resolution_clock::now();
    auto t_payoff =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
            .count();

    // Time just derivative computation
    t_start = std::chrono::high_resolution_clock::now();
    double payoff_delta_sum = 0.0;
    rng.seed(seed);
    for (std::size_t path = 0; path < benchmark_paths; ++path) {
      std::vector<double> normals_prefix(sim_per_path, 0.0);
      for (double &z : normals_prefix) {
        z = mcsim::CDF_inverse(unif(rng));
      }
      double factors_except_last = 1.0;
      for (std::size_t step = 0; step + 1 < sim_per_path; ++step) {
        const double factor = evolve_black_scholes_normal(rate, vol, dts[step],
                                                          normals_prefix[step]);
        factors_except_last *= factor;
      }
      double spot = g_last_step(spot0, factors_except_last, rate, vol,
                                dts.back(), normals_prefix.back());
      payoff_delta_sum += ad::forward_derivative<^^g_last_step, 0>(
                              spot0, factors_except_last, rate, vol, dts.back(),
                              normals_prefix.back()) *
                          ad::forward_derivative<PayoffFn, 0>(spot);
    }
    t_end = std::chrono::high_resolution_clock::now();
    auto t_derivative =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
            .count();

    // Time discontinuity analysis and correction
    t_start = std::chrono::high_resolution_clock::now();
    constexpr auto discontinuities =
        ad::get_discontinuity_points_and_amplitudes<PayoffFn, 0>();
    double correction_sum = 0.0;
    rng.seed(seed);
    for (std::size_t path = 0; path < benchmark_paths; ++path) {
      std::vector<double> normals_prefix(sim_per_path, 0.0);
      for (double &z : normals_prefix) {
        z = mcsim::CDF_inverse(unif(rng));
      }
      double factors_except_last = 1.0;
      for (std::size_t step = 0; step + 1 < sim_per_path; ++step) {
        const double factor = evolve_black_scholes_normal(rate, vol, dts[step],
                                                          normals_prefix[step]);
        factors_except_last *= factor;
      }

      for (std::size_t i = 0; i < discontinuities.size(); ++i) {
        const double z_star = ad::inverse_of_wrt<^^g_last_step, 5>(
            discontinuities.point(i), spot0, factors_except_last, rate, vol,
            dts.back());
        const double normal_pdf = mcsim::PDF(z_star);
        const double dg_d_spot0 = discontinuities.point(i) / spot0;
        const double dg_d_z = ad::forward_derivative<^^g_last_step, 5>(
            spot0, factors_except_last, rate, vol, dts.back(), z_star);
        const double inv_abs_dg_d_u = normal_pdf / std::abs(dg_d_z);
        correction_sum +=
            dg_d_spot0 * inv_abs_dg_d_u * discontinuities.amplitude(i);
      }
    }
    t_end = std::chrono::high_resolution_clock::now();
    auto t_correction =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
            .count();

    std::cout << label << " - " << benchmark_paths << " paths:\n";
    std::cout << "  Total time:             " << t_total << " ms\n";
    std::cout << "  Payoff computation:     " << t_payoff << " ms ("
              << (100.0 * t_payoff / t_total) << "%)\n";
    std::cout << "  Derivative computation: " << t_derivative << " ms ("
              << (100.0 * t_derivative / t_total) << "%)\n";
    std::cout << "  Correction term:        " << t_correction << " ms ("
              << (100.0 * t_correction / t_total) << "%)\n";
    std::cout << "  Time per path: " << (1000.0 * t_total / benchmark_paths)
              << " μs\n\n";
  };

  // Run timing benchmarks on a few payoff types
  run_timing_benchmark.template operator()<^^digital_call_payoff<100.0>>(
      "Digital Call", num_paths);

  run_timing_benchmark.template
  operator()<^^double_digital_payoff<99.0, 101.0>>("Double Digital", num_paths);

  run_timing_benchmark
      .template operator()<^^nonlinear_digital_call_payoff<100.0>>(
          "Nonlinear Digital Call", num_paths);

  return 0;
}
