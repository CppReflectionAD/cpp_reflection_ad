# Contributing

Everything here runs the same way locally and in CI. `run_tests.py` is the one
command for building the compilers and building and testing the repo; it drives
the [CMake presets](#cmake-presets), which CI calls directly. Where a new test
goes is in [tests/README.md](tests/README.md).

- [Prerequisites](#prerequisites)
- [Building the compilers](#building-the-compilers)
- [Building and testing](#building-and-testing)
- [CMake presets](#cmake-presets)
- [Debugging](#debugging)
- [Benchmarks](#benchmarks)
- [Formatting and linting](#formatting-and-linting)
- [CI checks](#ci-checks)
- [Bumping a compiler pin](#bumping-a-compiler-pin)

## Prerequisites

The tests need the two reflection compiler forks, which the repo builds from its
submodules (see [Building the compilers](#building-the-compilers)). Released
compilers can't build this code.

- **Linux** (CI uses Ubuntu 24.04): `git`, `cmake` (3.25 or newer), `ninja`,
  host `clang`/`clang++` and `ld.lld` to build clang, and for GCC the packages
  CI installs: `build-essential flex bison libgmp-dev libmpfr-dev libmpc-dev
  texinfo zlib1g-dev`.
- **macOS:** the Xcode command line tools (the GCC build finds the SDK with
  `xcrun`), plus `cmake` and `ninja`.
- **Python** 3.11 or newer. `run_tests.py` needs only the standard library.
  [uv](https://docs.astral.sh/uv/) runs the lint/format gate and the helper
  scripts in `tools/`.
- **A debugger:** `gdb` or `lldb`. In VS Code, CodeLLDB comes first (see
  [Debugging](#debugging)).

## Building the compilers

The repo is **self-contained**: `--build-compilers` builds clang from its own
`clang-p2996` submodule into `build/` — no externally pre-built compiler needed.
It checks out the submodule, builds clang (host `clang` + `lld`), builds the
Darwin `LTO` target as well on macOS so `libLTO.dylib` is present for the local
toolchain driver, builds the `libc++`/`libc++abi` runtimes (plus `libunwind`
except on macOS, where the system unwinder is used) with that clang, and syncs
the fork's `<meta>` header. It also builds what the sanitizer, coverage and
static-analysis checks need: compiler-rt's ASan, UBSan and profile runtimes,
`clang-tidy`, `llvm-cov`, `llvm-profdata` and `llvm-symbolizer`. On macOS it
first applies LLVM's fix for the sanitizers hanging at startup on macOS 26
(`tools/toolchain/patches/`) to the submodule, which then shows as modified in
`git status`.

GCC comes from the `gcc-mirror` submodule into `build/gcc-p2996`. On macOS it
is built from `gcc-darwin-reflect` instead
([CppReflectionAD/gcc-darwin-arm64](https://github.com/CppReflectionAD/gcc-darwin-arm64),
a fork of iains' Darwin port), with `gcc-mirror`'s reflection commits replayed
onto its pin first (see `--gcc-sync-*` in `run_tests.py --help`). The fork's
`p4395` branch is that pin with the same commits already applied, as of
`gcc-mirror` commit `812075e0`, and will replace the replay.
If the `gcc-mirror` pin has moved past that commit, `p4395` lacks the newer
commits; use the replay until the branch is updated.

Clones made before the submodule moved to the fork still fetch it from iains.
Update them once, after pulling:

```bash
git submodule sync gcc-darwin-reflect
```

To build `p4395` instead of the replay:

```bash
git submodule update --init gcc-darwin-reflect &&
  git -C gcc-darwin-reflect fetch origin p4395 &&
  git -C gcc-darwin-reflect checkout --detach origin/p4395 &&
  python3 run_tests.py --compiler gcc --build-compilers --no-gcc-sync
```

A later `--build-compilers` without `--no-gcc-sync` puts the tree back on the
pin and replays the commits, as before.

The recipes live in `tools/toolchain/`, and CI's toolchain cache runs the same
ones (`.github/scripts/build-*-toolchain.sh`), so a local build matches CI's.
CI builds clang itself as Release rather than RelWithDebInfo and strips the
result.

```bash
# one-time: build clang + runtimes into build/clang-p2996 (slow — builds LLVM)
python3 run_tests.py --compiler clang --build-compilers

# the same for GCC, into build/gcc-p2996
python3 run_tests.py --compiler gcc --build-compilers

# use a clang built elsewhere (or set CLANG_P2996_ROOT), or any compiler binary
python3 run_tests.py --compiler clang --clang-root /path/to/clang-p2996
python3 run_tests.py --compiler gcc --gcc-executable /path/to/g++

# build clang from another clang-p2996 checkout (or set REFLECT_CLANG_SOURCE_DIR),
# into its own clang root so the default build/clang-p2996 is left untouched
python3 run_tests.py --compiler clang --build-compilers \
    --clang-source-dir ../clang-p2996 \
    --clang-root build/clang-other --clang-runtimes-dir build/libcxx-other
```

## Building and testing

For each selected compiler (`--compiler clang`, `gcc` or `both`, the default)
and build type (`--build-type release`, the default, `debug`, `relwithdebinfo`,
or `all` for each in turn) `run_tests.py` configures the preset
(`clang-release`, ...) with that compiler, builds, and runs `ctest`, then
prints a summary. Everything else (the flags, which tests exist, the
`.fail.txt` comparison for `static_fail/` and `known_gap/` tests) is defined in
CMake, so `ctest` on its own gives the same results. Build trees go in
`build/cmake/<preset>`, shared with `cmake --preset` and VS Code; `-B DIR` puts
them in `DIR/<preset>` instead, and `--clean` deletes them first. Arguments
after `--` go to the CMake configure step.

```bash
# build and run every test (Release; the clang root defaults to build/clang-p2996)
python3 run_tests.py --compiler clang

# every build type, both compilers: the same presets CI runs
python3 run_tests.py --build-type all

# a single test, by ctest name or by path under tests/
python3 run_tests.py --compiler clang --tests ad_scalar_demo
python3 run_tests.py --compiler clang --tests 'common/ad_scalar_demo.cpp'

# every test under AddressSanitizer and UBSan (a Debug build, the
# <compiler>-sanitize preset); clang needs compiler-rt, which --build-compilers builds
python3 run_tests.py --sanitize
```

A `clang_only/` test asked for by name under `--compiler both` runs on clang
and is skipped, with a note, on GCC. `--verbose` shows the full build and
test commands.

Each demo prints its own `ALL CHECKS PASSED` / `ALL MATCH` on success. What
each test directory means, including the `static_fail/` and `known_gap/` tests
that pass by failing to compile, is in [tests/README.md](tests/README.md).

## CMake presets

`CMakePresets.json` has a configure, build and test preset per compiler and
build type: `clang-debug`, `clang-relwithdebinfo`, `clang-release`, and the same
for `gcc-`. Each one picks the compiler built in `build/` (`build/clang-p2996`,
`build/gcc-p2996`), uses Ninja, writes `compile_commands.json`, and builds into
`build/cmake/<preset>`.

`clang-sanitize` and `gcc-sanitize` are Debug builds with AddressSanitizer and
UBSan (`-DREFLECTION_AD_SANITIZE=ON`); a sanitizer report fails the test. On
GCC, six UBSan checks are left off, because their instrumentation changes what
reflection sees (`tests/known_gap/gcc/ubsan_reflection_repro.cpp`).

`clang-x86-64-v3` and `gcc-x86-64-v3` are Release builds with
`-march=x86-64-v3`, for x86-64 hosts only. That turns on FMA, which arm64 always
has, so CI on x86-64 also sees floating-point contraction bugs that otherwise
showed up only on arm64 Macs ([#95](https://github.com/CppReflectionAD/cpp_reflection_ad/pull/95)).

```bash
cmake --preset clang-debug           # configure
cmake --build --preset clang-debug   # build tests and benchmarks
ctest --preset clang-debug           # run the tests
```

To use a compiler from somewhere else, override it on the configure line
(`cmake --preset clang-debug -DCMAKE_CXX_COMPILER=/path/to/clang++`, as CI
does) or in a personal, git-ignored `CMakeUserPresets.json` that inherits from
these presets.

## Debugging

The binaries carry an rpath to their compiler's runtime library, so they need
no `LD_LIBRARY_PATH`. From the command line, either let `run_tests.py` build
one test in Debug and start it under a debugger:

```bash
python3 run_tests.py --compiler clang --tests ad_scalar_demo --debugger lldb
```

or run a built binary directly: `lldb build/cmake/clang-debug/tests/ad_scalar_demo`
(or `gdb ...`) works as is.

### VS Code

The repo checks in a VS Code setup in `.vscode/`: CMake Tools in presets mode,
debug configurations, and IntelliSense from `build/compile_commands.json`.

1. Build the compiler(s) (see [Building the compilers](#building-the-compilers));
   the presets look for them in `build/`.
2. Install the recommended extensions (VS Code offers them when the folder
   opens): `ms-vscode.cmake-tools` and `vadimcn.vscode-lldb` (CodeLLDB). For the
   `(cppdbg)` configurations, install `ms-vscode.cpptools` too.
3. In the CMake Tools side bar, pick a configure preset, e.g. `clang-p2996, Debug`
   (`clang-debug`).
4. To debug a test:
   - **From the Test Explorer:** right-click a test, then **Debug Test**. This
     uses the `Debug CTest test (lldb)` configuration.
   - **From Run and Debug (F5):** pick the test as the CMake launch target, then
     run `Debug launch target (lldb)` (or `(cppdbg)`).

The `(cppdbg)` configurations run gdb on Linux. On macOS, which has no gdb on
arm64, they run cpptools' `lldb-mi` with Apple's LLDB. Use them for GCC builds on
macOS: CodeLLDB's bundled lldb crashes showing a libstdc++ `std::unique_ptr` to a
class type. To make **Debug Test** use them, set `cmake.ctest.debugLaunchTarget`
to `Debug CTest test (cppdbg)`. To keep personal edits like that to the shared
`.vscode/` files out of `git status`, run `tools/vscode-local-edits.sh`
(`--undo` reverses it). The script's header says what to do when a pull changes
one of those files.

## Benchmarks

Benchmarks are built with the tests, but aren't registered with CTest. For
timings, build Release (the default) and run one, setting the workload with
`ITERATIONS`:

```bash
ITERATIONS=1000000 python3 run_tests.py --compiler clang --benchmarks black_scholes_timing --run-executables
```

The binaries are in `build/cmake/<preset>/benchmarks/`.

## Formatting and linting

[pre-commit](https://pre-commit.com) checks formatting (clang-format 21 for C++, ruff for Python), lint and whitespace. CI runs the same hooks. Python tooling is managed by [uv](https://docs.astral.sh/uv/):

```sh
uv run pre-commit run --all-files   # check and fix everything
uv run pre-commit install           # optional: run the hooks on every commit
```

The Python helper scripts' dependencies are in the `scripts` group, e.g. `uv run --group scripts tools/black_scholes_sympy_value_check.py`.

### Suppressing a finding

Fix the finding if you can. If not, suppress it as narrowly as possible, and
say why in a comment next to the suppression:

- **clang-format:** `// clang-format off` and `// clang-format on` around the
  lines to leave alone.
- **ruff:** `# noqa: <code>` at the end of the line, e.g. `# noqa: E731`. The
  rule set is spelled out in `[tool.ruff.lint]` in `pyproject.toml`.
- **The other pre-commit hooks:** an `exclude:` pattern on the hook in
  `.pre-commit-config.yaml`, as for the `.fail.txt` snapshots there.
- **Compiler warnings:** tests and benchmarks build with `-Wall -Wextra` as
  errors (`cmake/reflect_flags.cmake`). Around the code, use
  `#pragma GCC diagnostic push`, `#pragma GCC diagnostic ignored "-W<name>"`
  and `#pragma GCC diagnostic pop`, which both compilers honour. For a whole
  file, add `-Wno-<name>` to its `COMPILE_OPTIONS` in `tests/CMakeLists.txt`
  (see the `set_source_files_properties` calls there). To see every warning
  without failing the build, configure with
  `cmake --preset <preset> --compile-no-warning-as-error`.

## CI checks

Every pull request and every push to `main` runs these workflows, and the test
workflows also run nightly on `main` (03:17 UTC) to keep the toolchain caches
warm. Pull requests test the merge with `main`.

| Check | Workflow | Required | Reproduce locally |
|---|---|---|---|
| `pre-commit` | `pre-commit.yml` | yes | `uv lock --check && uv run pre-commit run --all-files` |
| `Clang tests` | `clang_prebuilt_cache.yml` | yes | `python3 run_tests.py --compiler clang --build-type all` |
| `GCC tests` | `gcc_prebuilt_cache.yml` | yes | `python3 run_tests.py --compiler gcc --build-type all` |
| `sanitize (ASan+UBSan)` | both test workflows | not yet ([#94](https://github.com/CppReflectionAD/cpp_reflection_ad/issues/94) C8) | `python3 run_tests.py --sanitize` |
| `test (Release, x86-64-v3)` | both test workflows | not yet | `cmake --preset clang-x86-64-v3 && cmake --build --preset clang-x86-64-v3 && ctest --preset clang-x86-64-v3` (or `gcc-`) |

`Clang tests` and `GCC tests` are summary jobs. Each passes only when its
workflow's `toolchain` job and its three `test (<build type>)` legs pass, one
per build type, so they match `--build-type all`.

The `toolchain` job restores the compiler from the Actions cache, or builds it
on a miss: about 1.5 hours for clang, about 30 minutes for GCC. The cache key
is the submodule's pinned commit plus a hash of
`.github/scripts/build-<compiler>-toolchain.sh`, `tools/toolchain/common.py`
and `tools/toolchain/<compiler>.py`. So a change to one of those files, or a
pin bump, rebuilds the toolchain, and a change to `common.py` rebuilds both.
A cache saved by a pull request can't be read from `main`, so the toolchain is
built again after merging. Workflow YAML isn't in the key.

## Bumping a compiler pin

The compilers are the submodules `clang-p2996` (the `expr-reflect` branch of
`siqiyyyy/clang-p2996`), `gcc-mirror` (the `autograd-capable` branch of
`yesmanchyk/gcc-mirror`) and, on macOS only, `gcc-darwin-reflect`. A bump is
its own pull request. For `clang-p2996` or `gcc-mirror`:

```bash
git -C clang-p2996 fetch origin expr-reflect
git -C clang-p2996 checkout --detach <commit>
git add clang-p2996

# rebuild the compiler from the new pin and run every test with it
python3 run_tests.py --compiler clang --build-compilers --build-type all
```

In the pull request, say what the bump brings in and compare the test results
of the old and new pins. Change `.gitmodules` only if the URL or branch
changes, and then run `git submodule sync` too.

CI rebuilds the toolchain on the pull request, and on `main` again after
merging (see [CI checks](#ci-checks)). Open pull requests rebuild it as well
once they test against the new `main`.

`gcc-darwin-reflect` is set to `ignore = all`, because the macOS build commits
`gcc-mirror`'s reflection commits onto it locally. So `git add` doesn't see a
new pin there; record it with
`git update-index --cacheinfo 160000,<commit>,gcc-darwin-reflect`. Its GCC
base must match `gcc-mirror`'s, so the two pins move together (see the comment
in `.gitmodules`).

Its URL is our fork of iains' repository, which has only the branches someone
pushed to it. So first push the new snapshot's branch from iains to the fork,
or `git submodule update` fails with "not our ref":

```bash
git -C gcc-darwin-reflect fetch https://github.com/iains/gcc-darwin-arm64.git <branch>
git -C gcc-darwin-reflect push origin FETCH_HEAD:refs/heads/<branch>
```
