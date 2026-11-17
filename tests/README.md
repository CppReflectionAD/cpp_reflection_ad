# Tests

Each directory holds one kind of test. Put a new test in the directory that
says which compilers it runs on and what it checks:

| Directory | What goes here | Passes when |
|---|---|---|
| `common/` | Tests that run under **both** compilers | it compiles and runs without a failed check |
| `clang_only/` | Tests that need the clang fork | as above, on clang only |
| `gcc_only/` | Tests that need the GCC fork | as above, on GCC only |
| `static_fail/` | Code that must be **rejected**: by the engine (a `static_assert` it should trip), or by every compiler because the standard makes it ill-formed | it fails to compile with the errors in its `.fail.txt` |
| `known_gap/clang/`, `known_gap/gcc/` | Code a compiler fork **should** accept but can't yet (a bug or missing feature in the fork), or accepts but should reject (then a `static_assert` fails only because it was accepted) | it fails to compile on that compiler with the errors in its `.fail.txt` |

`fixtures/` is not a test directory. It holds headers that tests (and
benchmarks) share: the functions to differentiate (`functions/`), the
Black-Scholes Monte Carlo model (`mc_sim/`), and helpers for tests that inspect
reflected function bodies (`reflection/`). Include them as
`"functions/1-poly.h"` or `"mc_sim/black_scholes.hpp"`.

`install/` isn't one either. It holds the `install_consumer` test: it installs
the build into a scratch prefix, then builds and runs `install/consumer/`, a
separate project that uses the library only through
`find_package(reflection_ad)`. Don't add tests there.

## Adding a test

- Add a `.cpp` file to the right directory. There is nothing to register:
  `CMakeLists.txt` finds it (re-run CMake, or `run_tests.py`, to pick it up).
  A `.cpp` anywhere else under `tests/` stops the configure.
- Run just that test with `python3 run_tests.py --tests <name>` (add
  `--debugger lldb` or `gdb` to start it under a debugger).
- Include the engine by its path under `include/`, e.g.
  `#include <ad/autograd.h>`, and the checks from `test_simple_include.hpp`
  (`EXPECT_EQUAL`, `EXPECT_NEAR_REL`, ...).
- The ctest name is the file name up to its first `.` (`cx_std.t.cpp` is
  `cx_std`), so it must be unique across `common/`, `clang_only/`,
  `gcc_only/` and `known_gap/`.
- A `static_fail/` or `known_gap/` test needs a `.fail.txt` snapshot of its
  expected errors; generate it with the `rebase-...` target. See
  [test_simple/README.md](../test_simple/README.md).

If the other compiler already accepts a `known_gap/` test, don't copy it into
`*_only/` for that compiler. Put `// TEST-PASSES-ON: clang` (or `gcc`) in the
first 10 lines instead: both `ctest` and `run_tests.py` then build and run it
as an ordinary test on that compiler, so one file checks the same asserts on
both.

When a known gap is fixed, the `known_gap/` test starts failing. What to do
depends on the kind of gap:

- **Valid code the compiler rejected:** the file now compiles. Move it to
  `common/` or the right `*_only/` directory, turn its expected failures into
  checks, drop any `TEST-PASSES-ON` line, and delete its `.fail.txt`.
- **Ill-formed code the compiler accepted:** the compiler now rejects it, so
  its errors no longer match the `.fail.txt`. Move the file to `static_fail/`,
  delete the `static_assert` that tripped only because the code was accepted,
  keep the comment naming the wording that makes the code ill-formed, and
  regenerate the `.fail.txt` on both compilers. Don't delete it: it is the
  check that another compiler, or a later pin, gets this case right. Any valid
  code next to it goes to `common/`.
