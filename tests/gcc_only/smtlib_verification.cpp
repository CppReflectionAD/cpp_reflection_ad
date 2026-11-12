// SMT-LIB for formally verifying functions, generated from their reflected
// bodies.
//
// smt::define_fun turns each function below into an SMT-LIB define-fun
// (tests/fixtures/smt/smtlib.hpp). The scripts put one or two of them next
// to a claim, negated, and ask the solver for a counterexample:
//   - median3.smt2: median3 returns the median of its arguments. unsat.
//   - max_equivalence.smt2: max_branchy and max_branchless agree on every
//     input. unsat.
//   - abs_nonnegative.smt2: abs_naive never returns a negative number. sat:
//     -INT_MIN wraps to INT_MIN, and the model says x = #x80000000.
// The test runs Z3 on the scripts and checks the results; run with a
// directory argument, it also writes the scripts there. See
// docs/smt-verification.md.
#include <filesystem>
#include <fstream>
#include <meta>
#include <string>
#include <string_view>

#include "smt/smtlib.hpp"
#include "test_simple_include.hpp"

int median3(int a, int b, int c) {
  if (a < b) {
    if (b < c)
      return b;
    if (a < c)
      return c;
    return a;
  }
  if (a < c)
    return a;
  if (b < c)
    return c;
  return b;
}

int max_branchy(int a, int b) {
  int m = a;
  if (b > m)
    m = b;
  return m;
}

int max_branchless(int a, int b) { return a > b ? a : b; }

// Not `x < 0 ? -x : x`, which GCC folds to its abs built-in before
// reflection sees it.
int abs_naive(int x) {
  if (x < 0)
    return -x;
  return x;
}

constexpr std::string_view max_branchy_smt =
    std::define_static_string(smt::define_fun(^^max_branchy));

// An assignment is a let, and the if merges m with an ite on its condition.
static_assert(max_branchy_smt ==
              "(define-fun max_branchy ((a (_ BitVec 32)) (b (_ BitVec 32))) "
              "(_ BitVec 32)\n"
              "  (let ((m!1 a))\n"
              "  (let ((m!2 b))\n"
              "  (let ((m!3 (ite (bvsgt b m!1) m!2 m!1)))\n"
              "  (let ((ret!4 m!3))\n"
              "  ret!4)))))\n");

constexpr std::string_view max_branchless_smt =
    std::define_static_string(smt::define_fun(^^max_branchless));

static_assert(max_branchless_smt ==
              "(define-fun max_branchless ((a (_ BitVec 32)) (b (_ BitVec "
              "32))) (_ BitVec 32)\n"
              "  (let ((ret!1 (ite (bvsgt a b) a b)))\n"
              "  ret!1))\n");

// -x is bvneg, which wraps.
constexpr std::string_view abs_naive_smt =
    std::define_static_string(smt::define_fun(^^abs_naive));

static_assert(abs_naive_smt.find("(bvneg x)") != std::string_view::npos);

constexpr std::string_view median3_smt =
    std::define_static_string(smt::define_fun(^^median3));

// Every return is guarded by whether an earlier one has happened.
static_assert(median3_smt.starts_with(
    "(define-fun median3 ((a (_ BitVec 32)) (b (_ BitVec 32)) (c (_ BitVec "
    "32))) (_ BitVec 32)\n"));
static_assert(median3_smt.find("(ite returned!") != std::string_view::npos);

// Claims to verify, written in C++ and translated via smt::define_fun.
int median3_claim(int a, int b, int c) {
  int r = median3(a, b, c);
  int le_count = (a <= r) + (b <= r) + (c <= r);
  int ge_count = (a >= r) + (b >= r) + (c >= r);
  return (r == a || r == b || r == c) && le_count >= 2 && ge_count >= 2;
}

int max_equivalence_claim(int a, int b) {
  return max_branchy(a, b) == max_branchless(a, b);
}

int abs_nonnegative_claim(int x) {
  return abs_naive(x) >= 0;
}

constexpr std::string_view median3_script = std::define_static_string(
    smt::refute_script({^^median3}, ^^median3_claim));

constexpr std::string_view max_equivalence_script = std::define_static_string(
    smt::refute_script({^^max_branchy, ^^max_branchless}, ^^max_equivalence_claim));

constexpr std::string_view abs_nonnegative_script = std::define_static_string(
    smt::refute_script({^^abs_naive}, ^^abs_nonnegative_claim, /*get_model=*/true));

struct script {
  const char *file;
  const char *expect;
  std::string_view text;
};

int main(int argc, char **argv) {
  const script scripts[] = {
      {"median3.smt2", "unsat", median3_script},
      {"max_equivalence.smt2", "unsat", max_equivalence_script},
      {"abs_nonnegative.smt2", "sat", abs_nonnegative_script},
  };

  // The functions still compute what the claims say, on a few inputs.
  EXPECT_EQUAL(median3(1, 2, 3), 2);
  EXPECT_EQUAL(median3(3, 1, 2), 2);
  EXPECT_EQUAL(median3(2, 3, 1), 2);
  EXPECT_EQUAL(median3(5, 5, 1), 5);
  EXPECT_TRUE(median3_claim(1, 2, 3));
  EXPECT_TRUE(median3_claim(3, 1, 2));
  EXPECT_TRUE(max_equivalence_claim(-4, 7));
  EXPECT_TRUE(abs_nonnegative_claim(-3));

  for (const script &s : scripts) {
    const std::string out = smt::z3(s.text);
    EXPECT_EQUAL(out.substr(0, out.find('\n')), s.expect);
    if (std::string_view(s.file) == "abs_nonnegative.smt2")
      EXPECT_TRUE(out.find("#x80000000") != std::string::npos);
    if (argc > 1) {
      std::filesystem::create_directories(argv[1]);
      std::ofstream f(std::string(argv[1]) + "/" + s.file);
      f << s.text;
      EXPECT_TRUE(static_cast<bool>(f));
    }
  }
  TEST_END;
}
