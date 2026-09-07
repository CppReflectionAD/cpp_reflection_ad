#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

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

double clamp_open_01(double u) {
  const double eps = std::numeric_limits<double>::epsilon();
  if (u <= eps)
    return eps;
  if (u >= 1.0 - eps)
    return 1.0 - eps;
  return u;
}

// Approximation due to Peter J. Acklam, adapted to C++.
double inverse_normal_cdf(double u) {
  constexpr double a1 = -39.69683028665376;
  constexpr double a2 = 220.9460984245205;
  constexpr double a3 = -275.9285104469687;
  constexpr double a4 = 138.3577518672690;
  constexpr double a5 = -30.66479806614716;
  constexpr double a6 = 2.506628277459239;

  constexpr double b1 = -54.47609879822406;
  constexpr double b2 = 161.5858368580409;
  constexpr double b3 = -155.6989798598866;
  constexpr double b4 = 66.80131188771972;
  constexpr double b5 = -13.28068155288572;

  constexpr double c1 = -0.007784894002430293;
  constexpr double c2 = -0.3223964580411365;
  constexpr double c3 = -2.400758277161838;
  constexpr double c4 = -2.549732539343734;
  constexpr double c5 = 4.374664141464968;
  constexpr double c6 = 2.938163982698783;

  constexpr double d1 = 0.007784695709041462;
  constexpr double d2 = 0.3224671290700398;
  constexpr double d3 = 2.445134137142996;
  constexpr double d4 = 3.754408661907416;

  constexpr double p_low = 0.02425;
  constexpr double p_high = 1.0 - p_low;

  const double p = clamp_open_01(u);

  if (p < p_low) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }

  if (p <= p_high) {
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  }

  const double q = std::sqrt(-2.0 * std::log(1.0 - p));
  return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
         ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

// One Black-Scholes step over dt, driven by a single uniform U(0,1) random
// variable.
double evolve_black_scholes(double spot, double r, double vol, double dt,
                            double uniform_u) {
  const double z = inverse_normal_cdf(uniform_u);
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

  const double call_price = monte_carlo_option_price(
      OptionType::Call, spot0, strike, rate, vol, dates, num_paths, seed);
  const double put_price = monte_carlo_option_price(
      OptionType::Put, spot0, strike, rate, vol, dates, num_paths, seed + 1);

  std::cout << "MC call price: " << call_price << "\n";
  std::cout << "MC put price : " << put_price << "\n";

  return 0;
}
