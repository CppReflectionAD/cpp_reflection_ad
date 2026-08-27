#include "autograd.h"
#include "functions/5-black_scholes.h"

#include <array>
#include <chrono>
#include <iostream>
// #include <limits>
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

  // Accumulators: [price, dS, dK, dv, dT]
  std::array<double, 5> fwd_avg{};
  std::array<double, 5> rev_avg{};

  // -----------------------------------------------------------------------
  // Forward-mode benchmark: gradient_of (4 forward passes per iteration)
  // -----------------------------------------------------------------------
  auto t0 = std::chrono::high_resolution_clock::now();
  for (std::size_t j = 0; j < iters; ++j) {
    double S = stock_distr(generator);
    double K = stock_distr(generator);
    double v = vol_distr(generator);
    double T = time_distr(generator);

    fwd_avg[0] += call_price(S, K, v, T);
    auto grad = ad::gradient_of<^^call_price>(S, K, v, T);
    fwd_avg[1] += grad[0]; // dS
    fwd_avg[2] += grad[1]; // dK
    fwd_avg[3] += grad[2]; // dv
    fwd_avg[4] += grad[3]; // dT
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto ms_fwd =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  // -----------------------------------------------------------------------
  // Reverse-mode benchmark: gradient_reverse (1 pass per iteration)
  // -----------------------------------------------------------------------
  // Re-seed so both benchmarks see the same inputs.
  generator.seed(123);
  auto t2 = std::chrono::high_resolution_clock::now();
  for (std::size_t j = 0; j < iters; ++j) {
    double S = stock_distr(generator);
    double K = stock_distr(generator);
    double v = vol_distr(generator);
    double T = time_distr(generator);

    rev_avg[0] += call_price(S, K, v, T);
    auto grad = ad::gradient_reverse<^^call_price>(S, K, v, T);
    rev_avg[1] += grad[0]; // dS
    rev_avg[2] += grad[1]; // dK
    rev_avg[3] += grad[2]; // dv
    rev_avg[4] += grad[3]; // dT
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  auto ms_rev =
      std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  // -----------------------------------------------------------------------
  // Report
  // -----------------------------------------------------------------------
  const double n = static_cast<double>(iters);
  std::cout.precision(std::numeric_limits<double>::max_digits10);

  std::cout << "=== Forward mode (gradient_of, " << iters
            << " iters) ===" << std::endl;
  std::cout << "  price  = " << fwd_avg[0] / n << std::endl;
  std::cout << "  dS     = " << fwd_avg[1] / n << std::endl;
  std::cout << "  dK     = " << fwd_avg[2] / n << std::endl;
  std::cout << "  dv     = " << fwd_avg[3] / n << std::endl;
  std::cout << "  dT     = " << fwd_avg[4] / n << std::endl;
  std::cout << "  time   = " << ms_fwd << " ms" << std::endl;

  std::cout << "=== Reverse mode (gradient_reverse, " << iters
            << " iters) ===" << std::endl;
  std::cout << "  price  = " << rev_avg[0] / n << std::endl;
  std::cout << "  dS     = " << rev_avg[1] / n << std::endl;
  std::cout << "  dK     = " << rev_avg[2] / n << std::endl;
  std::cout << "  dv     = " << rev_avg[3] / n << std::endl;
  std::cout << "  dT     = " << rev_avg[4] / n << std::endl;
  std::cout << "  time   = " << ms_rev << " ms" << std::endl;

  return 0;
}
