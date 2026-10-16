# Tests

Each directory holds one kind of test. Put a new test in the directory that
says which compilers it runs on and what it checks:

| Directory | What goes here | Passes when |
|---|---|---|
| `common/` | Tests that run under **both** compilers | it compiles and runs without a failed check |
| `clang_only/` | Tests that need the clang (`expr-reflect`) fork, e.g. anything using `include/ad/clang_only/` | as above, on clang only |
| `gcc_only/` | Tests that need the GCC fork | as above, on GCC only |
| `static_fail/` | Code the engine must **reject** (a `static_assert` it should trip) | it fails to compile with the errors in its `.fail.txt` |
| `known_gap/clang/`, `known_gap/gcc/` | Code a compiler fork **should** accept but can't yet (a bug or missing feature in the fork) | it fails to compile on that compiler with the errors in its `.fail.txt` |

`fixtures/` is not a test directory. It holds headers that tests (and
benchmarks) share: the functions to differentiate (`functions/`) and the
Black-Scholes Monte Carlo model (`mc_sim/`). Include them as
`"functions/1-poly.h"` or `"mc_sim/black_scholes.hpp"`.

## Adding a test

- Add a `.cpp` file to the right directory. There is nothing to register:
  `CMakeLists.txt` finds it (re-run CMake to pick it up), and so does
  `run_tests.py`, except that it skips `known_gap/` and only checks that a
  `static_fail/` test fails to compile; the `.fail.txt` comparison is `ctest`
  only. A `.cpp` anywhere else under `tests/` stops the configure.
- Include the engine by its path under `include/`, e.g.
  `#include <ad/autograd.h>`, and the checks from `test_simple_include.hpp`
  (`EXPECT_EQUAL`, `EXPECT_NEAR_REL`, ...).
- The ctest name is the file name up to its first `.` (`cx_std.t.cpp` is
  `cx_std`), so it must be unique across `common/`, `clang_only/` and
  `gcc_only/`.
- A `static_fail/` or `known_gap/` test needs a `.fail.txt` snapshot of its
  expected errors; generate it with the `rebase-...` target. See
  [test_simple/README.md](../test_simple/README.md).

When a known gap is fixed, the `known_gap/` test starts failing (the file
compiles). Move the file to `common/` or the right `*_only/` directory, turn
its expected failures into checks, and delete its `.fail.txt`.
