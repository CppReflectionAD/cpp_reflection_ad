#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "../tests/clang_only/discontinuity_analysis.hpp"
#include "../tests/clang_only/is_continuous.hpp"
#include "../tests/forward_derivative.h"
#include "../tests/is_invertible.hpp"
#include "../tests/mc_sim/black_scholes.hpp"
#include "../tests/mc_sim/evolve_black_scholes.hpp"

// first case: digital call option
double digital_call_payoff(double spot, double strike) {
  return (spot > strike) ? 1.0 : 0.0;
}

double digital_call_closed_form(double spot0, double strike, double rate,
                                double vol, double maturity_years) {
  // black_scholes.hpp formulas are expressed on forward variables.
  const double forward = spot0 * std::exp(rate * maturity_years);
  const double discount = std::exp(-rate * maturity_years);
  return discount * digital_call_price(forward, strike, vol, maturity_years);
}

// second case: digital AND call
double digital_and_call_payoff(double spot, double strike) {
  return (spot > strike) ? (1.0 + spot - strike) : 0.0;
}

double digital_and_call_closed_form(double spot0, double strike, double rate,
                                    double vol, double maturity_years) {
  const double forward = spot0 * std::exp(rate * maturity_years);
  const double discount = std::exp(-rate * maturity_years);
  return discount * (digital_call_price(forward, strike, vol, maturity_years) +
                     call_price(forward, strike, vol, maturity_years));
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

template <std::meta::info FinalPayoffFn>
std::array<double, 3>
monte_carlo_engine(double spot0, double strike, double r, double vol,
                   double maturity, std::size_t num_paths,
                   std::size_t sim_per_path, std::uint32_t seed) {
  if (spot0 <= 0.0 || strike <= 0.0 || vol <= 0.0 || maturity <= 0.0 ||
      num_paths == 0 || sim_per_path == 0) {
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

  constexpr bool payoff_is_continuous = ad::is_continuous_on<FinalPayoffFn>(
      ad::Interval{0.0, 1000000.0}, ad::Interval{100.0, 100.0});

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
    payoff_sum += static_cast<double>([:FinalPayoffFn:](spot, strike));

    double const spot_d = ad::forward_derivative<^^g_last_step, 0>(
        spot0, factors_except_last, r, vol, dts.back(), normals_prefix.back());
    payoff_delta_sum +=
        spot_d * ad::forward_derivative<FinalPayoffFn, 0>(spot, strike);

    if constexpr (!payoff_is_continuous) {
      // Tweak the last draw so terminal spot lands exactly on strike, using
      // the generic inverse machinery rather than an explicit closed form.
      const double target_factor = strike / spot_before_last;
      const double z_star =
          ad::inverse_of_wrt<^^evolve_black_scholes_normal, 3, double>(
              target_factor, r, vol, dts.back());
      const double normal_pdf = mcsim::PDF(z_star);

      // Sifting term written through the normal draw z:
      // contribution = f_Z(z*) * (dg/dS0)/|dg/dz| with f_Z = phi.
      // Since u = Phi(z), this is equivalent to using |dg/du| in U-space.
      const double dg_d_spot0 = strike / spot0;
      const double dg_d_z = ad::forward_derivative<^^g_last_step, 5>(
          spot0, factors_except_last, r, vol, dts.back(), z_star);
      const double inv_abs_dg_d_u = normal_pdf / std::abs(dg_d_z);

      correction_sum += dg_d_spot0 * inv_abs_dg_d_u;
    }
  }

  return {discount * (payoff_sum / static_cast<double>(num_paths)),
          discount * (payoff_delta_sum / static_cast<double>(num_paths)),
          discount * (correction_sum / static_cast<double>(num_paths))};
}

int main() {
  const TimePoint start{};
  const TimePoint maturity = start + Days{367};

  const double spot0 = 100.0;
  const double strike = 100.0;
  const double rate = 0.03;
  const double vol = 0.20;
  std::size_t num_paths = 200000;
  std::size_t sim_per_path = 1;
  if (auto *env_p = std::getenv("PATHS")) {
    num_paths = std::stoul(env_p);
  }
  if (auto *env_spp = std::getenv("SIM_PER_PATH")) {
    sim_per_path = std::stoul(env_spp);
  }

  if (num_paths == 0) {
    std::cerr << "PATHS must be a positive integer" << std::endl;
    return 1;
  }
  if (sim_per_path == 0) {
    std::cerr << "SIM_PER_PATH must be a positive integer" << std::endl;
    return 1;
  }

  const std::uint32_t seed = 42;
  const double maturity_years = year_fraction_act365(start, maturity);

  // Compile-time test: extract discontinuity points from digital_call_payoff
  // Expected: [100.0] (discontinuity at spot = strike = 100.0)
  constexpr auto digital_call_discontinuities =
      ad::get_discontinuity_points<^^digital_call_payoff, 0>(100.0);
  std::cout << "=== Compile-time Discontinuity Analysis ===\n";
  std::cout
      << "digital_call_payoff(x, K) discontinuity with K=100.0 found at x="
      << digital_call_discontinuities[0] << "\n\n";

  // digital call
  {
    const double cf_pv =
        digital_call_closed_form(spot0, strike, rate, vol, maturity_years);
    const double cf_delta =
        ad::forward_derivative<^^digital_call_closed_form, 0>(
            spot0, strike, rate, vol, maturity_years);
    const auto [mc_pv, mc_delta, mc_correction] =
        monte_carlo_engine<^^digital_call_payoff>(spot0, strike, rate, vol,
                                                  maturity_years, num_paths,
                                                  sim_per_path, seed);

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    std::cout << "Closed-form (digital call) : " << cf_pv << "\n";
    std::cout << "Closed-form delta          : " << cf_delta << "\n";
    std::cout << "MC digital call price      : " << mc_pv << "\n";
    std::cout << "MC Delta without correction: " << mc_delta << "\n";
    std::cout << "MC Correction term         : " << mc_correction << "\n";
    std::cout << "MC Delta                   : " << mc_delta + mc_correction
              << "\n";
  }

  // digital and call
  {
    const double cf_pv =
        digital_and_call_closed_form(spot0, strike, rate, vol, maturity_years);
    const double cf_delta =
        ad::forward_derivative<^^digital_and_call_closed_form, 0>(
            spot0, strike, rate, vol, maturity_years);
    const auto [mc_pv, mc_delta, mc_correction] =
        monte_carlo_engine<^^digital_and_call_payoff>(spot0, strike, rate, vol,
                                                      maturity_years, num_paths,
                                                      sim_per_path, seed);

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    std::cout << "Closed-form (digital call) : " << cf_pv << "\n";
    std::cout << "Closed-form delta          : " << cf_delta << "\n";
    std::cout << "MC digital call price      : " << mc_pv << "\n";
    std::cout << "MC Delta without correction: " << mc_delta << "\n";
    std::cout << "MC Correction term         : " << mc_correction << "\n";
    std::cout << "MC Delta                   : " << mc_delta + mc_correction
              << "\n";
  }

  return 0;
}
