// Verification conditions from loop invariants, for every input.
//
// smt::verify(^^f, ^^pre, ^^post) (tests/fixtures/smt/smtlib.hpp) replaces
// each loop of f by the invariant written in it with SMT_INVARIANT(...), and
// gives a script with one check per verification condition: the invariant
// holds on entry and is preserved, no `+ - *` in f overflows, and f returns a
// result that satisfies post. Every check is unsat when f is correct.
//   - sum_to.smt2, sum_to_for.smt2: 2 * sum_to(n) == n * (n + 1) for
//     0 <= n <= 65535, the largest n for which the sum fits in an int. All
//     unsat.
//   - sum_to_overflow.smt2: the same for n up to 65536. The overflow check
//     of s + i is sat: 1 + ... + 65536 doesn't fit.
//   - sum_to_wrong_invariant.smt2: an invariant that is false on entry. The
//     entry check is sat.
//   - sum_to_weak_invariant.smt2: an invariant that leaves out i <= n + 1,
//     which is true but too weak to give the result. The post check is sat.
//   - sum_capped.smt2: a loop with a break, whose result is 21 once n >= 6.
//     All unsat.
// Recognizing GCC's lowered loops needs is_goto_statement, is_label_statement
// and label_of; with a compiler that lacks them, the test only says it
// skipped. Run with a directory argument, it writes the scripts there. See
// docs/smt-verification.md.
#include <cstdio>
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
    SMT_INVARIANT(1 <= i && i <= n + 1 && 2 * s == i * (i - 1));
    s += i;
    ++i;
  }
  return s;
}

int sum_to_for(int n) {
  int s = 0;
  for (int i = 1; i <= n; ++i) {
    SMT_INVARIANT(1 <= i && i <= n + 1 && 2 * s == i * (i - 1));
    s += i;
  }
  return s;
}

int sum_to_wrong_invariant(int n) {
  int s = 0;
  int i = 1;
  while (i <= n) {
    SMT_INVARIANT(2 * s == i * i);
    s += i;
    ++i;
  }
  return s;
}

int sum_to_weak_invariant(int n) {
  int s = 0;
  int i = 1;
  while (i <= n) {
    SMT_INVARIANT(1 <= i && 2 * s == i * (i - 1));
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
    SMT_INVARIANT(1 <= i && i <= n + 1 && 2 * s == i * (i - 1) && s < 20);
    s += i;
    if (s >= 20)
      break;
    ++i;
  }
  return s;
}

// The specifications: mathematics, never run.
constexpr bool sum_pre(int n) { return 0 <= n && n <= 65535; }
constexpr bool sum_pre_too_big(int n) { return 0 <= n && n <= 65536; }
constexpr bool sum_post(int n, int r) { return 2 * r == n * (n + 1); }
constexpr bool capped_post(int n, int r) {
  return (n >= 6 && r == 21) || (n < 6 && 2 * r == n * (n + 1));
}

// smt::verify, dependent on D so that it is only evaluated where loops are
// supported.
template <class D>
consteval std::string vcs(std::meta::info fn, std::meta::info pre,
                          std::meta::info post) {
  return smt::verify(fn, pre, post);
}

struct script {
  const char *file;
  const char *expect;
  std::string_view text;
};

template <bool Supported, class D = void> struct invariant_scripts {
  static void run(int, char **) {
    std::printf("skipped: the compiler can't reflect gotos and labels\n");
  }
};

template <class D> struct invariant_scripts<true, D> {
  static constexpr std::string_view sum_to_vcs =
      std::define_static_string(vcs<D>(^^sum_to, ^^sum_pre, ^^sum_post));

  // One check each for entry, preservation, the overflows of s + i and
  // i + 1, and post.
  static_assert(sum_to_vcs.find("sum_to: the loop invariant holds on entry") !=
                std::string_view::npos);
  static_assert(sum_to_vcs.find("sum_to: the loop invariant is preserved") !=
                std::string_view::npos);
  static_assert(sum_to_vcs.find("sum_to: no overflow in (+ s!") !=
                std::string_view::npos);
  static_assert(sum_to_vcs.find("sum_to: no overflow in (+ i!") !=
                std::string_view::npos);
  static_assert(sum_to_vcs.find("sum_to: the function returns") !=
                std::string_view::npos);

  static void run(int argc, char **argv) {
    const script scripts[] = {
        {"sum_to.smt2", "all unsat", sum_to_vcs},
        {"sum_to_for.smt2", "all unsat",
         std::define_static_string(
             vcs<D>(^^sum_to_for, ^^sum_pre, ^^sum_post))},
        {"sum_to_overflow.smt2", "sat at the overflow of s + i",
         std::define_static_string(
             vcs<D>(^^sum_to, ^^sum_pre_too_big, ^^sum_post))},
        {"sum_to_wrong_invariant.smt2", "sat at entry",
         std::define_static_string(
             vcs<D>(^^sum_to_wrong_invariant, ^^sum_pre, ^^sum_post))},
        {"sum_to_weak_invariant.smt2", "sat at post",
         std::define_static_string(
             vcs<D>(^^sum_to_weak_invariant, ^^sum_pre, ^^sum_post))},
        {"sum_capped.smt2", "all unsat",
         std::define_static_string(
             vcs<D>(^^sum_capped, ^^sum_pre, ^^capped_post))},
    };
    for (const script &s : scripts) {
      std::printf(";; ==== %s (expect %s)\n%.*s\n", s.file, s.expect,
                  static_cast<int>(s.text.size()), s.text.data());
      if (argc > 1) {
        std::ofstream out(std::string(argv[1]) + "/" + s.file);
        out << s.text;
        EXPECT_TRUE(static_cast<bool>(out));
      }
    }
  }
};

int main(int argc, char **argv) {
  // The functions compute what the specifications say, on a few inputs; the
  // invariants don't run.
  EXPECT_EQUAL(sum_to(0), 0);
  EXPECT_EQUAL(sum_to(10), 55);
  EXPECT_EQUAL(sum_to(65535), 2147450880);
  EXPECT_EQUAL(sum_to_for(10), 55);
  EXPECT_EQUAL(sum_to_wrong_invariant(10), 55);
  EXPECT_EQUAL(sum_to_weak_invariant(10), 55);
  EXPECT_EQUAL(sum_capped(4), 10);
  EXPECT_EQUAL(sum_capped(100), 21);
  static_assert(sum_post(10, 55) && capped_post(100, 21) && sum_pre(65535));

  invariant_scripts<smt::loops_supported>::run(argc, argv);
  TEST_END;
}
