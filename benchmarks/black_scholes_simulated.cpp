#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "../tests/mc_sim/balck_scholes.hpp"
#include "../tests/mc_sim/normal_distribution.hpp"

enum class OptionType { Call, Put };

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
  for (auto d = start + step; d < end; d += step)
    dates.push_back(d);
  if (dates.back() != end)
    dates.push_back(end);
  return dates;
}

// One Black-Scholes step over dt, driven by a single uniform U(0,1) random
// variable.
double evolve_black_scholes(double spot, double r, double vol, double dt,
                            double uniform_u) {
  const double z = mcsim::CDF_inverse(uniform_u);
  const double drift = (r - 0.5 * vol * vol) * dt;
  const double diffusion = vol * std::sqrt(dt) * z;
  return spot * std::exp(drift + diffusion);
}

double monte_carlo_option_price(OptionType type, double spot0, double strike,
                                double r, double vol,
                                const std::vector<TimePoint> &dates,
                                std::uint32_t num_paths, std::uint32_t seed) {
  if (dates.size() < 2 || num_paths == 0)
    return 0.0;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  const double maturity = year_fraction_act365(dates.front(), dates.back());
  const double discount = std::exp(-r * maturity);

  double payoff_sum = 0.0;
  for (std::uint32_t path = 0; path < num_paths; ++path) {
    double spot = spot0;
    for (std::size_t i = 1; i < dates.size(); ++i) {
      const double dt = year_fraction_act365(dates[i - 1], dates[i]);
      spot = evolve_black_scholes(spot, r, vol, dt, unif(rng));
    }

    const double intrinsic =
        (type == OptionType::Call) ? (spot - strike) : (strike - spot);
    payoff_sum += std::max(0.0, intrinsic);
  }

  return discount * (payoff_sum / static_cast<double>(num_paths));
}

int main() {
  const TimePoint start{};
  const TimePoint maturity = start + Days{367};
  const auto dates = build_date_grid(start, maturity, Days{30});

  const double spot0 = 100.0;
  const double strike = 100.0;
  const double rate = 0.03;
  const double vol = 0.20;
  const std::uint32_t num_paths = 200000;
  const std::uint32_t seed = 42;
  const double maturity_years = year_fraction_act365(start, maturity);

  const double mc_call_price = monte_carlo_option_price(
      OptionType::Call, spot0, strike, rate, vol, dates, num_paths, seed);
  const double mc_put_price = monte_carlo_option_price(
      OptionType::Put, spot0, strike, rate, vol, dates, num_paths, seed + 1);

  // The helper in balck_scholes.hpp is written on forward variables.
  const double forward = spot0 * std::exp(rate * maturity_years);
  const double discount = std::exp(-rate * maturity_years);
  const double closed_form_call =
      discount * call_price(forward, strike, vol, maturity_years);
  const double closed_form_put =
      discount * put_price(forward, strike, vol, maturity_years);

  std::cout << "MC call price      : " << mc_call_price << "\n";
  std::cout << "MC put price       : " << mc_put_price << "\n";
  std::cout << "Closed-form (call) : " << closed_form_call << "\n";
  std::cout << "Closed-form (put)  : " << closed_form_put << "\n";

  return 0;
}
