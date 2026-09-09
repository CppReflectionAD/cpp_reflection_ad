// TEST-FLAGS: -O2
// TEST-FLAGS-CLANG: -fconstexpr-steps=16000000
// TEST-FLAGS-GCC: -fconstexpr-ops-limit=64000000

#include "../tests/forward_derivative.h"
#include "../tests/higher_order_taylor_ad.h"
#include "../tests/recursive_higher_order_derivative.h"
#include "../tests/reverse_derivative.h"

#include "functions/5-black_scholes.h"

#include <array>
#include <chrono>
#include <iostream>
#include <random>

int main() {
  std::mt19937 generator(123);
  std::uniform_real_distribution<double> stock_distr(90.0, 110.0);
  std::uniform_real_distribution<double> vol_distr(0.05, 0.3);
  std::uniform_real_distribution<double> time_distr(0.5, 1.5);

  std::size_t iters = 10000;
  if (auto *env_p = std::getenv("ITERATIONS")) {
    iters = std::stoul(env_p);
  }

  if (iters == 0) {
    std::cerr << "ITERATIONS must be a positive integer" << std::endl;
    return 1;
  }

  // Accumulators: [dS, dK, dv, dT]
  std::array<double, 4> gradient_of_avg{};
  std::array<double, 4> gradient_reverse_avg{};
  double price_avg = 0;

  // Accumulators: [ds, d2ds, dtds d3ds]
  std::array<double, 4> recursive_higher_order_avg{};
  // Accumulators: first 5 derivatives wrt S, and then the 10th/20th
  std::array<double, 7> taylor_higher_order_avg{};

  // Runs `body(S, K, v, T)` for `iters` random draws and returns the elapsed
  // time in milliseconds. The generator is re-seeded each time so that every
  // benchmark sees exactly the same inputs.
  auto benchmark = [&](auto &&body) {
    generator.seed(123);
    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t j = 0; j < iters; ++j) {
      double S = stock_distr(generator);
      double K = stock_distr(generator);
      double v = vol_distr(generator);
      double T = time_distr(generator);

      body(S, K, v, T);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
  };

  // --------------------------------------------------------------------
  // Forward-mode benchmark: gradient_of (4 forward passes per iteration)
  // --------------------------------------------------------------------
  auto ms_fwd = benchmark([&](double S, double K, double v, double T) {
    auto grad = ad::gradient_of<^^call_price>(S, K, v, T);
    gradient_of_avg[0] += grad[0]; // dS
    gradient_of_avg[1] += grad[1]; // dK
    gradient_of_avg[2] += grad[2]; // dv
    gradient_of_avg[3] += grad[3]; // dT
  });

  // -----------------------------------------------------------------------
  // Reverse-mode benchmark: gradient_reverse (1 pass per iteration)
  // -----------------------------------------------------------------------
  auto ms_rev = benchmark([&](double S, double K, double v, double T) {
    auto grad = ad::gradient_reverse<^^call_price>(S, K, v, T);
    gradient_reverse_avg[0] += grad[0]; // dS
    gradient_reverse_avg[1] += grad[1]; // dK
    gradient_reverse_avg[2] += grad[2]; // dv
    gradient_reverse_avg[3] += grad[3]; // dT
  });

  // -----------------------------------------------------------------------
  // Benchmark the function call itself
  // -----------------------------------------------------------------------
  auto ms_fn = benchmark([&](double S, double K, double v, double T) {
    price_avg += call_price(S, K, v, T);
  });

  // -----------------------------------------------------------------------
  // Benchmark the function call itself
  // -----------------------------------------------------------------------
  auto ms_recursive_dds =
      benchmark([&](double S, double K, double v, double T) {
        recursive_higher_order_avg[0] +=
            ad::partial_derivative<^^call_price, 0>(S, K, v, T);
      });

  auto ms_recursive_d2ds =
      benchmark([&](double S, double K, double v, double T) {
        recursive_higher_order_avg[1] +=
            ad::partial_derivative<^^call_price, 0, 0>(S, K, v, T);
      });

  auto ms_recursive_dtds =
      benchmark([&](double S, double K, double v, double T) {
        recursive_higher_order_avg[2] +=
            ad::partial_derivative<^^call_price, 0, 3>(S, K, v, T);
      });

  auto ms_recursive_d3ds =
      benchmark([&](double S, double K, double v, double T) {
        recursive_higher_order_avg[3] +=
            ad::partial_derivative<^^call_price, 0, 0, 0>(S, K, v, T);
      });

  // -----------------------------------------------------------------------
  // Benchmark the function call itself
  // -----------------------------------------------------------------------
  auto ms_taylor_dds = benchmark([&](double S, double K, double v, double T) {
    taylor_higher_order_avg[0] +=
        ad::taylor_mode_ad<^^call_price, 0>(S, K, v, T);
  });

  auto ms_taylor_d2ds = benchmark([&](double S, double K, double v, double T) {
    taylor_higher_order_avg[1] +=
        ad::taylor_mode_ad<^^call_price, 0, 0>(S, K, v, T);
  });

  auto ms_taylor_d3ds = benchmark([&](double S, double K, double v, double T) {
    taylor_higher_order_avg[2] +=
        ad::taylor_mode_ad<^^call_price, 0, 0, 0>(S, K, v, T);
  });

  auto ms_taylor_d4ds = benchmark([&](double S, double K, double v, double T) {
    taylor_higher_order_avg[3] +=
        ad::taylor_mode_ad<^^call_price, 0, 0, 0, 0>(S, K, v, T);
  });

  auto ms_taylor_d5ds = benchmark([&](double S, double K, double v, double T) {
    taylor_higher_order_avg[4] +=
        ad::taylor_mode_ad<^^call_price, 0, 0, 0, 0, 0>(S, K, v, T);
  });

  auto ms_taylor_d10ds = benchmark([&](double S, double K, double v, double T) {
    taylor_higher_order_avg[5] +=
        ad::taylor_mode_ad<^^call_price, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(S, K, v,
                                                                       T);
  });

  auto ms_taylor_d20ds = benchmark([&](double S, double K, double v, double T) {
    taylor_higher_order_avg[6] +=
        ad::taylor_mode_ad<^^call_price, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0>(S, K, v, T);
  });

  // ------
  // Report
  // ------
  const double n = static_cast<double>(iters);
  std::cout.precision(std::numeric_limits<double>::max_digits10);

  std::cout << "=== Forward mode (gradient_of, " << iters
            << " iters) ===" << std::endl;
  std::cout << "  dS     = " << gradient_of_avg[0] / n << std::endl;
  std::cout << "  dK     = " << gradient_of_avg[1] / n << std::endl;
  std::cout << "  dv     = " << gradient_of_avg[2] / n << std::endl;
  std::cout << "  dT     = " << gradient_of_avg[3] / n << std::endl;
  std::cout << "  time   = " << ms_fwd << " ms" << std::endl;

  std::cout << "=== Reverse mode (gradient_reverse, " << iters
            << " iters) ===" << std::endl;
  std::cout << "  dS     = " << gradient_reverse_avg[0] / n << std::endl;
  std::cout << "  dK     = " << gradient_reverse_avg[1] / n << std::endl;
  std::cout << "  dv     = " << gradient_reverse_avg[2] / n << std::endl;
  std::cout << "  dT     = " << gradient_reverse_avg[3] / n << std::endl;
  std::cout << "  time   = " << ms_rev << " ms" << std::endl;

  std::cout << "=== Recursive Higher orders (partial_derivative, " << iters
            << " iters) ===" << std::endl;
  std::cout << "  dS     = " << recursive_higher_order_avg[0] / n << std::endl;
  std::cout << "  time   = " << ms_recursive_dds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  dS2    = " << recursive_higher_order_avg[1] / n << std::endl;
  std::cout << "  time   = " << ms_recursive_d2ds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  dTdS   = " << recursive_higher_order_avg[2] / n << std::endl;
  std::cout << "  time   = " << ms_recursive_dtds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  dS3    = " << recursive_higher_order_avg[3] / n << std::endl;
  std::cout << "  time   = " << ms_recursive_d3ds << " ms" << std::endl;

  std::cout << "=== Taylor Higher orders (taylor_mode_ad, " << iters
            << " iters) ===" << std::endl;
  std::cout << "  dS     = " << taylor_higher_order_avg[0] / n << std::endl;
  std::cout << "  time   = " << ms_taylor_dds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  dS2    = " << taylor_higher_order_avg[1] / n << std::endl;
  std::cout << "  time   = " << ms_taylor_d2ds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  d3S    = " << taylor_higher_order_avg[2] / n << std::endl;
  std::cout << "  time   = " << ms_taylor_d3ds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  d4S    = " << taylor_higher_order_avg[3] / n << std::endl;
  std::cout << "  time   = " << ms_taylor_d4ds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  d5S    = " << taylor_higher_order_avg[4] / n << std::endl;
  std::cout << "  time   = " << ms_taylor_d5ds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  d10S    = " << taylor_higher_order_avg[5] / n << std::endl;
  std::cout << "  time   = " << ms_taylor_d10ds << " ms" << std::endl;
  std::cout << "  ---------------------------------" << std::endl;
  std::cout << "  d20S    = " << taylor_higher_order_avg[6] / n << std::endl;
  std::cout << "  time   = " << ms_taylor_d20ds << " ms" << std::endl;

  std::cout << "=== Evaluation of the primary function (" << iters
            << " iters) ===" << std::endl;
  std::cout << "  Price  = " << price_avg / n << std::endl;
  std::cout << "  time   = " << ms_fn << " ms" << std::endl;

  return 0;
}
