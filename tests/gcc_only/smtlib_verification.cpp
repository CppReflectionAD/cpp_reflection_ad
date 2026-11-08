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

constexpr std::string_view max_branchless_smt =
    std::define_static_string(smt::define_fun(^^max_branchless));

constexpr std::string_view abs_naive_smt =
    std::define_static_string(smt::define_fun(^^abs_naive));

constexpr std::string_view median3_smt =
    std::define_static_string(smt::define_fun(^^median3));

struct script {
  const char *file;
  const char *expect;
  std::string text;
};

std::string median3_script() {
  return std::string("; median3 returns the median of a, b and c: at least\n"
                     "; two of them are <= it and at least two are >= it.\n"
                     "(set-logic QF_BV)\n") +
         std::string(median3_smt) +
         "(declare-const a (_ BitVec 32))\n"
         "(declare-const b (_ BitVec 32))\n"
         "(declare-const c (_ BitVec 32))\n"
         "(define-fun r () (_ BitVec 32) (median3 a b c))\n"
         "(define-fun one ((p Bool)) (_ BitVec 2) (ite p #b01 #b00))\n"
         "(define-fun le_count () (_ BitVec 2)\n"
         "  (bvadd (one (bvsle a r)) (one (bvsle b r)) (one (bvsle c r))))\n"
         "(define-fun ge_count () (_ BitVec 2)\n"
         "  (bvadd (one (bvsge a r)) (one (bvsge b r)) (one (bvsge c r))))\n"
         "(assert (not (and (or (= r a) (= r b) (= r c))\n"
         "                  (bvuge le_count #b10)\n"
         "                  (bvuge ge_count #b10))))\n"
         "(check-sat)\n";
}

std::string max_equivalence_script() {
  return std::string("; max_branchy and max_branchless agree on every input.\n"
                     "(set-logic QF_BV)\n") +
         std::string(max_branchy_smt) + std::string(max_branchless_smt) +
         "(declare-const a (_ BitVec 32))\n"
         "(declare-const b (_ BitVec 32))\n"
         "(assert (not (= (max_branchy a b) (max_branchless a b))))\n"
         "(check-sat)\n";
}

std::string abs_nonnegative_script() {
  return std::string("; abs_naive never returns a negative number. It does:\n"
                     "; -INT_MIN wraps to INT_MIN.\n"
                     "(set-logic QF_BV)\n") +
         std::string(abs_naive_smt) +
         "(declare-const x (_ BitVec 32))\n"
         "(assert (not (bvsge (abs_naive x) #x00000000)))\n"
         "(check-sat)\n"
         "(get-value (x))\n";
}

int main(int argc, char **argv) {
  const script scripts[] = {
      {"median3.smt2", "unsat", median3_script()},
      {"max_equivalence.smt2", "unsat", max_equivalence_script()},
      {"abs_nonnegative.smt2", "sat", abs_nonnegative_script()},
  };

  // The functions still compute what the scripts claim, on a few inputs.
  EXPECT_EQUAL(median3(1, 2, 3), 2);
  EXPECT_EQUAL(median3(3, 1, 2), 2);
  EXPECT_EQUAL(median3(2, 3, 1), 2);
  EXPECT_EQUAL(median3(5, 5, 1), 5);
  EXPECT_EQUAL(max_branchy(-4, 7), max_branchless(-4, 7));
  EXPECT_EQUAL(abs_naive(-3), 3);

  for (const script &s : scripts) {
    const std::string out = smt::z3(s.text);
    EXPECT_EQUAL(out.substr(0, out.find('\n')), s.expect);
    if (std::string_view(s.file) == "abs_nonnegative.smt2")
      EXPECT_TRUE(out.find("#x80000000") != std::string::npos);
    if (argc > 1) {
      std::ofstream f(std::string(argv[1]) + "/" + s.file);
      f << s.text;
      EXPECT_TRUE(static_cast<bool>(f));
    }
  }
  TEST_END;
}
