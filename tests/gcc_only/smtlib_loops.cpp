// SMT-LIB for verifying functions with loops, by bounded unrolling.
//
// smt::define_fun(^^f, k) unrolls each loop of f k times and also defines
// f_unwound, true when k iterations were enough (tests/fixtures/smt/
// smtlib.hpp). Each script below assumes a precondition that bounds the
// loop, and asks for an input where f_unwound is false or the claim is: unsat
// proves the claim for every input the precondition allows.
//   - sum_to.smt2, sum_to_for.smt2: sum_to(n) == n(n+1)/2, by a while and by a
//     for, for 0 <= n <= 15. unsat.
//   - sum_to_off_by_one.smt2: the same claim for a loop that stops at n - 1.
//     sat, with a counterexample.
//   - sum_to_bound.smt2: sum_to's claim for 0 <= n <= 20 with 16
//     unrollings, which aren't enough. sat: sum_to_unwound is false.
//   - sum_capped.smt2 (break), sum_skip3.smt2 (continue), countdown_do.smt2
//     (do/while): closed forms for 0 <= n <= 15. unsat.
// Recognizing GCC's lowered loops needs is_goto_statement, is_label_statement
// and label_of; with a compiler that lacks them (smt::loops_supported is
// false), the test only says it skipped. Run with a directory argument, it
// writes the scripts there. See docs/smt-verification.md.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <meta>
#include <string>
#include <string_view>

#include "smt/smtlib.hpp"
#include "test_simple_include.hpp"

int sum_to(int n) {
  int s = 0;
  int i = 1;
  while (i <= n) {
    s += i;
    ++i;
  }
  return s;
}

int sum_to_for(int n) {
  int s = 0;
  for (int i = 1; i <= n; ++i)
    s += i;
  return s;
}

int sum_to_off_by_one(int n) {
  int s = 0;
  int i = 1;
  while (i < n) {
    s += i;
    ++i;
  }
  return s;
}

// 1 + 2 + ... until the sum reaches 20, or up to n.
int sum_capped(int n) {
  int s = 0;
  int i = 1;
  while (i <= n) {
    s += i;
    if (s >= 20)
      break;
    ++i;
  }
  return s;
}

// 1 + 2 + ... + n, leaving out 3.
int sum_skip3(int n) {
  int s = 0;
  for (int i = 1; i <= n; ++i) {
    if (i == 3)
      continue;
    s += i;
  }
  return s;
}

// n + (n - 1) + ... + 1, or just n when n <= 0: the body runs once.
int countdown_do(int n) {
  int s = 0;
  do {
    s += n;
    --n;
  } while (n > 0);
  return s;
}

constexpr int unroll = 16;

// smt::define_fun of FN, dependent on D so that it is only evaluated where
// loops are supported.
template <class D> consteval std::string smt_of(std::meta::info fn) {
  return smt::define_fun(fn, unroll);
}

constexpr const char *prelude = "(set-logic QF_BV)\n";

// n(n+1)/2, for the claims.
constexpr const char *triangle =
    "(declare-const n (_ BitVec 32))\n"
    "(define-fun tri ((k (_ BitVec 32))) (_ BitVec 32)\n"
    "  (bvsdiv (bvmul k (bvadd k #x00000001)) #x00000002))\n";

std::string bounded(int max_n) {
  char buf[96];
  std::snprintf(buf, sizeof buf,
                "(assert (and (bvsle #x00000000 n) (bvsle n #x%08x)))\n",
                max_n);
  return buf;
}

// `(assert (not (and (F_unwound n) CLAIM)))`: some input breaks the claim or
// needs more unrolling.
std::string refute(const char *f, const std::string &claim) {
  return std::string("(assert (not (and (") + f + "_unwound n) " + claim +
         ")))\n(check-sat)\n";
}

struct script {
  const char *file;
  const char *expect;
  std::string text;
};

template <bool Supported, class D = void> struct loop_scripts {
  static void run(int, char **) {
    std::printf("skipped: the compiler can't reflect gotos and labels\n");
  }
};

template <class D> struct loop_scripts<true, D> {
  static constexpr std::string_view sum_to_smt =
      std::define_static_string(smt_of<D>(^^sum_to));
  static constexpr std::string_view sum_to_for_smt =
      std::define_static_string(smt_of<D>(^^sum_to_for));
  static constexpr std::string_view off_by_one_smt =
      std::define_static_string(smt_of<D>(^^sum_to_off_by_one));
  static constexpr std::string_view sum_capped_smt =
      std::define_static_string(smt_of<D>(^^sum_capped));
  static constexpr std::string_view sum_skip3_smt =
      std::define_static_string(smt_of<D>(^^sum_skip3));
  static constexpr std::string_view countdown_do_smt =
      std::define_static_string(smt_of<D>(^^countdown_do));

  static void run(int argc, char **argv) {
    const std::string tri_claim = "(= (sum_to n) (tri n))";
    const script scripts[] = {
        {"sum_to.smt2", "unsat",
         std::string("; sum_to(n) == n(n+1)/2 for 0 <= n <= 15.\n") + prelude +
             std::string(sum_to_smt) + triangle + bounded(15) +
             refute("sum_to", tri_claim)},
        {"sum_to_for.smt2", "unsat",
         std::string("; The same, written with a for.\n") + prelude +
             std::string(sum_to_for_smt) + triangle + bounded(15) +
             refute("sum_to_for", "(= (sum_to_for n) (tri n))")},
        {"sum_to_off_by_one.smt2", "sat",
         std::string("; A loop that stops at n - 1 misses the last term.\n") +
             prelude + std::string(off_by_one_smt) + triangle + bounded(15) +
             refute("sum_to_off_by_one", "(= (sum_to_off_by_one n) (tri n))") +
             "(get-value (n (sum_to_off_by_one n)))\n"},
        {"sum_to_bound.smt2", "sat",
         std::string("; 16 unrollings aren't enough for n up to 20: the\n"
                     "; solver finds an n where sum_to_unwound is false.\n") +
             prelude + std::string(sum_to_smt) + triangle + bounded(20) +
             refute("sum_to", tri_claim) +
             "(get-value (n (sum_to_unwound n)))\n"},
        {"sum_capped.smt2", "unsat",
         std::string("; The break: the sum stops at 21 = 1 + ... + 6 once\n"
                     "; it reaches 20.\n") +
             prelude + std::string(sum_capped_smt) + triangle + bounded(15) +
             refute("sum_capped", "(= (sum_capped n) (ite (bvsge n #x00000006)"
                                  " #x00000015 (tri n)))")},
        {"sum_skip3.smt2", "unsat",
         std::string("; The continue: 3 is left out once n reaches it.\n") +
             prelude + std::string(sum_skip3_smt) + triangle + bounded(15) +
             refute("sum_skip3", "(= (sum_skip3 n) (bvsub (tri n) (ite (bvsge"
                                 " n #x00000003) #x00000003 #x00000000)))")},
        {"countdown_do.smt2", "unsat",
         std::string("; The do/while runs its body once even when n <= 0.\n") +
             prelude + std::string(countdown_do_smt) + triangle +
             "(assert (and (bvsle #xfffffff6 n) (bvsle n #x0000000f)))\n" +
             refute("countdown_do", "(= (countdown_do n) (ite (bvsge n "
                                    "#x00000001) (tri n) n))")},
    };
    for (const script &s : scripts) {
      const std::string out = smt::z3(s.text);
      EXPECT_EQUAL(out.substr(0, out.find('\n')), s.expect);
      if (std::string_view(s.file) == "sum_to_bound.smt2")
        EXPECT_TRUE(out.find("((sum_to_unwound n) false)") !=
                    std::string::npos);
      if (argc > 1) {
        std::filesystem::create_directories(argv[1]);
        std::ofstream f(std::string(argv[1]) + "/" + s.file);
        f << s.text;
        EXPECT_TRUE(static_cast<bool>(f));
      }
    }
  }
};

int main(int argc, char **argv) {
  // The functions compute what the scripts claim, on a few inputs.
  EXPECT_EQUAL(sum_to(0), 0);
  EXPECT_EQUAL(sum_to(10), 55);
  EXPECT_EQUAL(sum_to_for(10), 55);
  EXPECT_EQUAL(sum_to_off_by_one(10), 45);
  EXPECT_EQUAL(sum_capped(4), 10);
  EXPECT_EQUAL(sum_capped(10), 21);
  EXPECT_EQUAL(sum_skip3(2), 3);
  EXPECT_EQUAL(sum_skip3(5), 12);
  EXPECT_EQUAL(countdown_do(4), 10);
  EXPECT_EQUAL(countdown_do(-3), -3);

  loop_scripts<smt::loops_supported>::run(argc, argv);
  TEST_END;
}
