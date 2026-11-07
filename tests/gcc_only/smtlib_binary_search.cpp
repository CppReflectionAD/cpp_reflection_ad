// Binary search, verified for every sorted array and key.
//
// smt::verify (tests/fixtures/smt/smtlib.hpp) treats the pointer parameter as
// an array, checks every a[i] in the code to be within it, and reads
// smt::forall in the specifications and invariants as a quantifier. Every
// check is unsat when the function is correct.
//   - binary_search.smt2: binary_search returns an index of key, or -1 when
//     key isn't in the array, for every sorted array of n >= 0 ints. All
//     unsat.
//   - binary_search_overflow.smt2: the midpoint as (lo + hi) / 2, the bug
//     that sat in the JDK's binary search for nine years. The overflow check
//     of lo + hi is sat: lo + hi passes INT_MAX for arrays of over 2^30
//     ints.
//   - binary_search_weak_invariant.smt2: an invariant that says nothing
//     about the elements from hi on. The post check is sat.
//   - binary_search_out_of_bounds.smt2: a loop condition of lo <= hi with hi
//     starting at n, which reads a[n]. The bounds check of a[mid] is sat.
// Recognizing GCC's lowered loops needs is_goto_statement, is_label_statement
// and label_of, and reading a[i] needs subscripts reflected as written; with
// a compiler that lacks label_of, the test only says it skipped. Run with a
// directory argument, it writes the scripts there. See
// docs/smt-verification.md.
#include <cstdio>
#include <fstream>
#include <meta>
#include <string>
#include <string_view>

#include "smt/smtlib.hpp"
#include "test_simple_include.hpp"

int binary_search(const int *a, int n, int key) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    smt::invariant([&] {
      return 0 <= lo && lo <= hi && hi <= n &&
             smt::forall(0, lo, [&](int k) { return a[k] < key; }) &&
             smt::forall(hi, n, [&](int k) { return a[k] > key; });
    });
    int mid = lo + (hi - lo) / 2;
    if (a[mid] < key)
      lo = mid + 1;
    else if (a[mid] > key)
      hi = mid;
    else
      return mid;
  }
  return -1;
}

int binary_search_overflow(const int *a, int n, int key) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    smt::invariant([&] {
      return 0 <= lo && lo <= hi && hi <= n &&
             smt::forall(0, lo, [&](int k) { return a[k] < key; }) &&
             smt::forall(hi, n, [&](int k) { return a[k] > key; });
    });
    int mid = (lo + hi) / 2;
    if (a[mid] < key)
      lo = mid + 1;
    else if (a[mid] > key)
      hi = mid;
    else
      return mid;
  }
  return -1;
}

int binary_search_weak_invariant(const int *a, int n, int key) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    smt::invariant([&] {
      return 0 <= lo && lo <= hi && hi <= n &&
             smt::forall(0, lo, [&](int k) { return a[k] < key; });
    });
    int mid = lo + (hi - lo) / 2;
    if (a[mid] < key)
      lo = mid + 1;
    else if (a[mid] > key)
      hi = mid;
    else
      return mid;
  }
  return -1;
}

int binary_search_out_of_bounds(const int *a, int n, int key) {
  int lo = 0;
  int hi = n;
  while (lo <= hi) {
    smt::invariant([&] {
      return 0 <= lo && lo <= hi + 1 && hi <= n &&
             smt::forall(0, lo, [&](int k) { return a[k] < key; }) &&
             smt::forall(hi + 1, n, [&](int k) { return a[k] > key; });
    });
    int mid = lo + (hi - lo) / 2;
    if (a[mid] < key)
      lo = mid + 1;
    else if (a[mid] > key)
      hi = mid - 1;
    else
      return mid;
  }
  return -1;
}

// a holds n ints in ascending order: each is <= every later one.
constexpr bool search_pre(const int *a, int n, [[maybe_unused]] int key) {
  return 0 <= n && smt::valid_range(a, n) && smt::forall(0, n, [&](int i) {
           return smt::forall(i, n, [&](int j) { return a[i] <= a[j]; });
         });
}

// r is an index of key, or -1 when no element is key.
constexpr bool search_post(const int *a, int n, int key, int r) {
  return (0 <= r && r < n && a[r] == key) ||
         (r == -1 && smt::forall(0, n, [&](int k) { return a[k] != key; }));
}

// smt::verify, dependent on D so that it is only evaluated where loops are
// supported.
template <class D> consteval std::string vcs(std::meta::info fn) {
  return smt::verify(fn, ^^search_pre, ^^search_post);
}

struct script {
  const char *file;
  const char *expect;
  std::string_view text;
};

template <bool Supported, class D = void> struct search_scripts {
  static void run(int, char **) {
    std::printf("skipped: the compiler can't reflect gotos and labels\n");
  }
};

template <class D> struct search_scripts<true, D> {
  static constexpr std::string_view vcs_ok =
      std::define_static_string(vcs<D>(^^binary_search));

  // The array and its length, a quantifier, and a bounds check of a[mid].
  static_assert(vcs_ok.find("(set-logic ALL)") != std::string_view::npos);
  static_assert(vcs_ok.find("(declare-const a (Array Int Int))") !=
                std::string_view::npos);
  static_assert(vcs_ok.find("(forall ((") != std::string_view::npos);
  static_assert(vcs_ok.find("binary_search: index in bounds: a[mid!") !=
                std::string_view::npos);

  static void run(int argc, char **argv) {
    const script scripts[] = {
        {"binary_search.smt2", "all unsat", vcs_ok},
        {"binary_search_overflow.smt2", "sat at the overflow of lo + hi",
         std::define_static_string(vcs<D>(^^binary_search_overflow))},
        {"binary_search_weak_invariant.smt2", "sat at post",
         std::define_static_string(vcs<D>(^^binary_search_weak_invariant))},
        {"binary_search_out_of_bounds.smt2", "sat at the bounds of a[mid]",
         std::define_static_string(vcs<D>(^^binary_search_out_of_bounds))},
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
  const int a[] = {1, 3, 3, 7, 10, 42};
  const int n = 6;
  EXPECT_TRUE(search_pre(a, n, 7));
  for (int key = 0; key <= 43; ++key) {
    const int r = binary_search(a, n, key);
    EXPECT_TRUE(search_post(a, n, key, r));
    EXPECT_EQUAL(binary_search_overflow(a, n, key), r);
    EXPECT_EQUAL(binary_search_weak_invariant(a, n, key), r);
  }
  EXPECT_EQUAL(binary_search(a, n, 7), 3);
  EXPECT_EQUAL(binary_search(a, n, 8), -1);
  EXPECT_EQUAL(binary_search(a, 0, 1), -1);

  search_scripts<smt::loops_supported>::run(argc, argv);
  TEST_END;
}
