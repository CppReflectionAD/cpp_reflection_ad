#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "../tests/mc_sim/black_scholes.hpp"
#include "../tests/mc_sim/evolve_black_scholes.hpp"

using TimePoint = std::chrono::system_clock::time_point;
using Days = std::chrono::duration<std::int64_t, std::ratio<86400>>;

double year_fraction_act365(const TimePoint &from, const TimePoint &to)
{
    const double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(to - from)
            .count();
    return seconds / (365.0 * 24.0 * 3600.0);
}

std::vector<TimePoint> build_date_grid(const TimePoint &start,
                                       const TimePoint &end,
                                       Days step = Days{30})
{
    std::vector<TimePoint> dates;
    dates.push_back(start);
    for (auto d = start + step; d < end; d += step)
    {
        dates.push_back(d);
    }

    if (dates.back() != end)
    {
        dates.push_back(end);
    }

    return dates;
}

double monte_carlo_digital_call_price(double spot0, double strike, double r,
                                      double vol,
                                      const std::vector<TimePoint> &dates,
                                      std::uint32_t num_paths,
                                      std::uint32_t seed)
{
    if (dates.size() < 2 || num_paths == 0)
    {
        return 0.0;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    const double maturity = year_fraction_act365(dates.front(), dates.back());
    const double discount = std::exp(-r * maturity);

    double payoff_sum = 0.0;
    for (std::uint32_t path = 0; path < num_paths; ++path)
    {
        double spot = spot0;
        for (std::size_t i = 1; i < dates.size(); ++i)
        {
            const double dt = year_fraction_act365(dates[i - 1], dates[i]);
            spot = evolve_black_scholes(spot, r, vol, dt, unif(rng));
        }

        payoff_sum += (spot > strike) ? 1.0 : 0.0;
    }

    return discount * (payoff_sum / static_cast<double>(num_paths));
}

int main()
{
    const TimePoint start{};
    const TimePoint maturity = start + Days{367};
    const auto dates = build_date_grid(start, maturity, Days{30});

    const double spot0 = 100.0;
    const double strike = 100.0;
    const double rate = 0.03;
    const double vol = 0.20;
    std::size_t num_paths = 200000;
    if (auto *env_p = std::getenv("PATHS"))
    {
        num_paths = std::stoul(env_p);
    }

    if (num_paths == 0)
    {
        std::cerr << "PATHS must be a positive integer" << std::endl;
        return 1;
    }

    const std::uint32_t seed = 42;
    const double maturity_years = year_fraction_act365(start, maturity);

    const double mc_digital_call_price = monte_carlo_digital_call_price(
        spot0, strike, rate, vol, dates, num_paths, seed);

    // The helper in black_scholes.hpp is written on forward variables.
    const double forward = spot0 * std::exp(rate * maturity_years);
    const double discount = std::exp(-rate * maturity_years);
    const double closed_form_digital_call =
        discount * digital_call_price(forward, strike, vol, maturity_years);

    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "MC digital call price      : " << mc_digital_call_price
              << "\n";
    std::cout << "Closed-form (digital call) : " << closed_form_digital_call
              << "\n";

    return 0;
}
