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
#include "../tests/is_invertible.hpp"
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

template <std::meta::info FinalPayoffFn>
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
            const double normal_z = mcsim::CDF_inverse(unif(rng));
            const double factor = evolve_black_scholes(r, vol, dt, normal_z);
            spot *= factor;
        }

        payoff_sum += static_cast<double>([:FinalPayoffFn:](spot, strike));
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

double monte_carlo_discontinuity_delta_contribution(
    double spot0, double strike, double r, double vol, double maturity,
    std::size_t num_paths, std::size_t sim_per_path, std::uint32_t seed)
{
    if (spot0 <= 0.0 || strike <= 0.0 || vol <= 0.0 || maturity <= 0.0 ||
        num_paths == 0 || sim_per_path == 0)
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

    const double dt_last = dts.back();
    const double sqrt_dt_last = std::sqrt(dt_last);
    const double drift_last = (r - 0.5 * vol * vol) * dt_last;

    double correction_sum = 0.0;
    for (std::size_t path = 0; path < num_paths; ++path)
    {
        std::vector<double> normals(sim_per_path, 0.0);

        double spot_before_last = spot0;
        for (std::size_t step = 0; step + 1 < sim_per_path; ++step)
        {
            normals[step] = mcsim::CDF_inverse(unif(rng));
            const double factor =
                evolve_black_scholes(r, vol, dts[step], normals[step]);
            spot_before_last *= factor;
        }

        // Tweak the last draw so terminal spot lands exactly on strike, using
        // the generic inverse machinery rather than an explicit closed form.
        const double target_factor = strike / spot_before_last;
        normals.back() = ad::inverse_of_wrt<^^evolve_black_scholes, 3, double>(
            target_factor, r, vol, dt_last);
        const double z_star = normals.back();
        const double normal_pdf = mcsim::PDF(z_star);

        // Sifting term written through the normal draw z:
        // contribution = f_Z(z*) * (dg/dS0)/|dg/dz| with f_Z = phi.
        // Since u = Phi(z), this is equivalent to using |dg/du| in U-space.
        const double dg_d_spot0 = strike / spot0;
        const double d_factor_d_z =
            ad::forward_derivative<^^evolve_black_scholes, 3>(r, vol, dt_last,
                                                              z_star);
        const double dg_d_z = spot_before_last * d_factor_d_z;
        const double inv_abs_dg_d_u = normal_pdf / std::abs(dg_d_z);

        correction_sum += dg_d_spot0 * inv_abs_dg_d_u;
    }

    return discount * (correction_sum / static_cast<double>(num_paths));
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
        monte_carlo_digital_call_price<^^final_payoff>(
            spot0, strike, rate, vol, maturity_years, num_paths, sim_per_path,
            seed);

    const double closed_form_digital_call =
        closed_form_discounted_digital_call_price(spot0, strike, rate, vol,
                                                  maturity_years);
    const double d_price_delta =
        ad::forward_derivative<^^closed_form_discounted_digital_call_price, 0>(
            spot0, strike, rate, vol, maturity_years);
    const double d_price_delta_discontinuity =
        monte_carlo_discontinuity_delta_contribution(spot0, strike, rate, vol,
                                                     maturity_years, num_paths,
                                                     sim_per_path, seed);
    constexpr bool payoff_is_discontinuous = is_discontinuous<^^final_payoff>(
        ad::Interval{99.0, 101.0}, ad::Interval{100.0, 100.0});

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    std::cout << "Closed-form (digital call) : " << closed_form_digital_call
              << "\n";
    std::cout << "Closed-form delta          : " << d_price_delta << "\n";
    std::cout << "Delta discontinuity term   : " << d_price_delta_discontinuity
              << "\n";
    std::cout << "final_payoff discontinuous : " << payoff_is_discontinuous
              << "\n";
    std::cout << "MC digital call price      : " << mc_digital_call_price
              << "\n";
    return 0;
}
