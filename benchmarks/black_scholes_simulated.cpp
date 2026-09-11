#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "../tests/clang_only/is_continuous.hpp"
#include "../tests/forward_derivative.h"
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

double final_payoff(double spot, double strike)
{
    return (spot > strike) ? 1.0 : 0.0;
}

template <std::meta::info Fn, typename... Intervals>
consteval bool is_discontinuous(Intervals... bounds)
{
    return !ad::is_continuous_on<Fn>(bounds...);
}

double monte_carlo_digital_call_price(double spot0, double strike, double r,
                                      double vol, double maturity,
                                      std::size_t num_paths,
                                      std::size_t sim_per_path,
                                      std::uint32_t seed)
{
    if (maturity <= 0.0 || num_paths == 0 || sim_per_path == 0)
    {
        return 0.0;
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
    for (std::size_t path = 0; path < num_paths; ++path)
    {
        double spot = spot0;
        for (double dt : dts)
        {
            const double factor = evolve_black_scholes(r, vol, dt, unif(rng));
            spot *= factor;
        }

        payoff_sum += final_payoff(spot, strike);
    }

    return discount * (payoff_sum / static_cast<double>(num_paths));
}

double closed_form_discounted_digital_call_price(double spot0, double strike,
                                                 double rate, double vol,
                                                 double maturity_years)
{
    // black_scholes.hpp formulas are expressed on forward variables.
    const double forward = spot0 * std::exp(rate * maturity_years);
    const double discount = std::exp(-rate * maturity_years);
    return discount * digital_call_price(forward, strike, vol, maturity_years);
}

int main()
{
    const TimePoint start{};
    const TimePoint maturity = start + Days{367};

    const double spot0 = 100.0;
    const double strike = 100.0;
    const double rate = 0.03;
    const double vol = 0.20;
    std::size_t num_paths = 200000;
    std::size_t sim_per_path = 1;
    if (auto *env_p = std::getenv("PATHS"))
    {
        num_paths = std::stoul(env_p);
    }
    if (auto *env_spp = std::getenv("SIM_PER_PATH"))
    {
        sim_per_path = std::stoul(env_spp);
    }

    if (num_paths == 0)
    {
        std::cerr << "PATHS must be a positive integer" << std::endl;
        return 1;
    }
    if (sim_per_path == 0)
    {
        std::cerr << "SIM_PER_PATH must be a positive integer" << std::endl;
        return 1;
    }

    const std::uint32_t seed = 42;
    const double maturity_years = year_fraction_act365(start, maturity);

    const double mc_digital_call_price =
        monte_carlo_digital_call_price(spot0, strike, rate, vol, maturity_years,
                                       num_paths, sim_per_path, seed);

    const double closed_form_digital_call =
        closed_form_discounted_digital_call_price(spot0, strike, rate, vol,
                                                  maturity_years);
    const double d_price_delta =
        ad::forward_derivative<^^closed_form_discounted_digital_call_price, 0>(
            spot0, strike, rate, vol, maturity_years);
    constexpr bool payoff_is_discontinuous = is_discontinuous<^^final_payoff>(
        ad::Interval{99.0, 101.0}, ad::Interval{100.0, 100.0});

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    std::cout << "Closed-form (digital call) : " << closed_form_digital_call
              << "\n";
    std::cout << "Closed-form delta          : " << d_price_delta << "\n";
    std::cout << "final_payoff discontinuous : " << payoff_is_discontinuous
              << "\n";
    std::cout << "MC digital call price      : " << mc_digital_call_price
              << "\n";
    return 0;
}
