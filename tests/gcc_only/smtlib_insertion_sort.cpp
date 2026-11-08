// Insertion sort, verified for every array: the result is sorted and a
// permutation of the input.
//
// smt::verify (tests/fixtures/smt/smtlib.hpp) treats `a[j] = x` as a store,
// handles the inner loop inside the outer one, reads smt::old(a) as the array
// on entry, and smt::permutation as a relation with only the axioms every
// permutation satisfies: an array is one of itself, and a swap keeps one. So
// the proof needs the code to rearrange the array by swaps.
//   - insertion_sort.smt2: sorted and a permutation of the input, for every
//     array of n >= 0 ints. All unsat.
//   - insertion_sort_out_of_bounds.smt2: the inner loop runs to j >= 0, so it
//     reads a[-1]. The bounds checks of a[j - 1] are sat.
//   - insertion_sort_lost_element.smt2: a "swap" without the temporary,
//     which copies a[j - 1] over a[j]. Preservation of the inner invariant,
//     whose permutation it no longer follows from, isn't unsat.
//   - insertion_sort_weak_invariant.smt2: an inner invariant that leaves out
//     that a[j] is at most the elements after it. The preservation of the
//     outer invariant isn't unsat.
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

// a[lo, hi) is in ascending order: each element is at most every later one.
// (Every pair, not just neighbors: the solver doesn't do induction.)
constexpr bool sorted(const int *a, int lo, int hi) {
  return smt::forall(lo, hi, [&](int p) {
    return smt::forall(p, hi, [&](int q) { return a[p] <= a[q]; });
  });
}

void insertion_sort(int *a, int n) {
  for (int i = 1; i < n; ++i) {
    smt::invariant([&] {
      return 1 <= i && (i <= n || i == 1) && sorted(a, 0, i) &&
             smt::permutation(a, smt::old(a), 0, n);
    });
    for (int j = i; j > 0 && a[j - 1] > a[j]; --j) {
      // a[0, i] is sorted but for a[j], which is at most the elements after
      // it.
      smt::invariant([&] {
        return 0 <= j && j <= i && i < n &&
               smt::forall(
                   0, i + 1,
                   [&](int p) {
                     return smt::forall(p, i + 1, [&](int q) {
                       return p == j || q == j || a[p] <= a[q];
                     });
                   }) &&
               smt::forall(
                   j + 1, i + 1, [&](int q) { return a[j] <= a[q]; }) &&
               smt::permutation(a, smt::old(a), 0, n);
      });
      int t = a[j];
      a[j] = a[j - 1];
      a[j - 1] = t;
    }
  }
}

void insertion_sort_out_of_bounds(int *a, int n) {
  for (int i = 1; i < n; ++i) {
    smt::invariant([&] {
      return 1 <= i && (i <= n || i == 1) && sorted(a, 0, i) &&
             smt::permutation(a, smt::old(a), 0, n);
    });
    for (int j = i; j >= 0 && a[j - 1] > a[j]; --j) {
      smt::invariant([&] {
        return -1 <= j && j <= i && i < n &&
               smt::permutation(a, smt::old(a), 0, n);
      });
      int t = a[j];
      a[j] = a[j - 1];
      a[j - 1] = t;
    }
  }
}

void insertion_sort_lost_element(int *a, int n) {
  for (int i = 1; i < n; ++i) {
    smt::invariant([&] {
      return 1 <= i && (i <= n || i == 1) && sorted(a, 0, i) &&
             smt::permutation(a, smt::old(a), 0, n);
    });
    for (int j = i; j > 0 && a[j - 1] > a[j]; --j) {
      smt::invariant([&] {
        return 0 <= j && j <= i && i < n &&
               smt::forall(
                   0, i + 1,
                   [&](int p) {
                     return smt::forall(p, i + 1, [&](int q) {
                       return p == j || q == j || a[p] <= a[q];
                     });
                   }) &&
               smt::forall(
                   j + 1, i + 1, [&](int q) { return a[j] <= a[q]; }) &&
               smt::permutation(a, smt::old(a), 0, n);
      });
      a[j] = a[j - 1];
      a[j - 1] = a[j];
    }
  }
}

void insertion_sort_weak_invariant(int *a, int n) {
  for (int i = 1; i < n; ++i) {
    smt::invariant([&] {
      return 1 <= i && (i <= n || i == 1) && sorted(a, 0, i) &&
             smt::permutation(a, smt::old(a), 0, n);
    });
    for (int j = i; j > 0 && a[j - 1] > a[j]; --j) {
      smt::invariant([&] {
        return 0 <= j && j <= i && i < n &&
               smt::forall(
                   0, i + 1,
                   [&](int p) {
                     return smt::forall(p, i + 1, [&](int q) {
                       return p == j || q == j || a[p] <= a[q];
                     });
                   }) &&
               smt::permutation(a, smt::old(a), 0, n);
      });
      int t = a[j];
      a[j] = a[j - 1];
      a[j - 1] = t;
    }
  }
}

// The specifications. post takes the parameters as they were on entry, then
// the array as it is at the end.
constexpr bool sort_pre(const int *a, int n) {
  return 0 <= n && smt::valid_range(a, n);
}
constexpr bool sort_post(const int *before, int n, const int *a) {
  return sorted(a, 0, n) && smt::permutation(a, before, 0, n);
}

// smt::verify, dependent on D so that it is only evaluated where loops are
// supported.
template <class D> consteval std::string vcs(std::meta::info fn) {
  return smt::verify(fn, ^^sort_pre, ^^sort_post);
}

struct script {
  const char *file;
  const char *expect;
  std::string_view text;
};

template <bool Supported, class D = void> struct sort_scripts {
  static void run(int, char **) {
    std::printf("skipped: the compiler can't reflect gotos and labels\n");
  }
};

template <class D> struct sort_scripts<true, D> {
  static constexpr std::string_view vcs_ok =
      std::define_static_string(vcs<D>(^^insertion_sort));

  // The permutation axioms, a store for each write, and both loops'
  // invariants.
  static_assert(vcs_ok.find("(declare-fun perm ") != std::string_view::npos);
  static_assert(vcs_ok.find("(store ") != std::string_view::npos);
  static_assert(vcs_ok.find("insertion_sort: post holds when the function "
                            "returns") != std::string_view::npos);

  static void run(int argc, char **argv) {
    const script scripts[] = {
        {"insertion_sort.smt2", "all unsat", vcs_ok},
        {"insertion_sort_out_of_bounds.smt2", "sat at the bounds of a[j - 1]",
         std::define_static_string(vcs<D>(^^insertion_sort_out_of_bounds))},
        {"insertion_sort_lost_element.smt2",
         "not unsat at the inner preservation",
         std::define_static_string(vcs<D>(^^insertion_sort_lost_element))},
        {"insertion_sort_weak_invariant.smt2",
         "not unsat at the outer preservation",
         std::define_static_string(vcs<D>(^^insertion_sort_weak_invariant))},
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
  // The sort does what the specifications say, on a few arrays, and the
  // broken one loses an element.
  const int inputs[][6] = {{5, 2, 9, 1, 5, 6},
                           {1, 2, 3, 4, 5, 6},
                           {6, 5, 4, 3, 2, 1},
                           {7, 7, 7, 7, 7, 7}};
  for (const auto &input : inputs) {
    int a[6], b[6];
    for (int k = 0; k < 6; ++k)
      a[k] = b[k] = input[k];
    insertion_sort(a, 6);
    EXPECT_TRUE(sort_pre(input, 6));
    EXPECT_TRUE(sort_post(input, 6, a));
    insertion_sort_weak_invariant(b, 6);
    EXPECT_TRUE(sort_post(input, 6, b));
  }
  int lost[] = {3, 1, 2};
  insertion_sort_lost_element(lost, 3);
  const int lost_input[] = {3, 1, 2};
  EXPECT_FALSE(smt::permutation(lost, lost_input, 0, 3));
  int empty[1] = {0};
  insertion_sort(empty, 0);

  sort_scripts<smt::loops_supported>::run(argc, argv);
  TEST_END;
}
