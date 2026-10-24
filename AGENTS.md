# Agent instructions

[CONTRIBUTING.md](CONTRIBUTING.md) says how to build, test, debug and lint, and
[tests/README.md](tests/README.md) where a test goes. Read them first; this file
lists only the points that are easy to get wrong.

- **Only the forks understand this code.** Released clang, GCC, clangd and
  clang-tidy can't parse the reflection syntax (`^^`, `[: :]`), so ignore their
  diagnostics. Build through `run_tests.py` or a preset, never a bare compiler
  call or `cmake` without `--preset`: the flags come from CMake targets.
- **Don't build the compilers unless asked.** `--build-compilers` takes hours.
  Check for `build/clang-p2996/bin/clang++` and
  `build/gcc-p2996/artifacts/bin/g++` instead, and say if one is missing.
- **Iterate on one test** with `--tests <name>`, and run the full suite at the
  end.
- **Run the lint gate before committing.** It rewrites files, so check the diff
  afterwards.
- **Leave the toolchain recipe alone unless that's the task.** Any edit to
  `tools/toolchain/` or `.github/scripts/build-*-toolchain.sh` rebuilds the CI
  toolchains, which takes hours (see "CI checks" in CONTRIBUTING.md).
- **Don't commit submodule changes** except in a pin bump. On macOS,
  `clang-p2996` shows as modified after a compiler build; that is expected.
- **Reflect function bodies through `ad::meta`, not `std::meta`.** The GCC
  fork's `<meta>` still has the older expression-reflection API, and
  `include/ad/meta.h` supplies the clang fork's on top of it. Code outside
  `clang_only/` that calls `std::meta::operands_of`, `is_id_expression` and
  the like directly breaks every GCC build.
- **Constant evaluation quirks in the forks:**
  - `<cmath>` isn't `constexpr`. Use `include/cx_std/` (`cx::exp`, `log`,
    `sqrt`, `erfc`, `sin`, `cos`).
  - In a `constexpr` function, naming a `constexpr` local that holds a
    `std::meta::info` (such as a reflected node) in a runtime expression makes
    the function immediate, usually with no note saying why. Copy the fields
    you need into plain `constexpr` locals first.
  - In functions whose bodies are reflected, declare one variable per
    declaration: `const double a = 1, b = 2;` breaks the reflection.
  - clang rejects NaN results and division by zero in constant evaluation; GCC
    also rejects overflow to infinity.
  - Don't define a macro named `V`: libc++'s `<meta>` uses it as an identifier.
