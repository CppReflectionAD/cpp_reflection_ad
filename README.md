# cpp_reflection_ad

> [!WARNING]
> **`.vscode/` is now tracked ([#124](https://github.com/CppReflectionAD/cpp_reflection_ad/pull/124)).** Back up your local `.vscode/settings.json`,
> `launch.json` and `extensions.json` before you pull: git replaces ignored files
> with tracked ones without asking. Afterwards, copy your changes back and run
> `tools/vscode-local-edits.sh` so git leaves them alone. Details in [#125](https://github.com/CppReflectionAD/cpp_reflection_ad/issues/125).

Proof of concept for automatic differentiation of C++ code using static reflection. Based on (and hoping to extend) the C++ reflection proposal P2996R13:
https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r13.html

## Try it on Compiler Explorer (godbolt)

The `expr-reflect` clang fork this repo builds on
([siqiyyyy/clang-p2996, branch `expr-reflect`](https://github.com/siqiyyyy/clang-p2996/tree/expr-reflect))
is available on [Compiler Explorer](https://godbolt.org/) as
**x86-64 clang (expr-reflect)**, so you can try expression reflection and this
library in the browser without building a compiler. It is rebuilt nightly from
the tip of the `expr-reflect` branch, the same way Bloomberg's P2996 fork is
offered as "x86-64 clang (reflection - C++26)".

Example: <https://godbolt.org/z/z3nsh4GcW>

To use it:

1. Open <https://godbolt.org/>, choose C++ as the language, and pick
   **x86-64 clang (expr-reflect)** from the compiler list (it is in the
   experimental Clang x86-64 group; type `expr-reflect` in the compiler picker
   to find it).
2. The default options are
   `-std=c++26 -freflection -fexpansion-statements -stdlib=libc++`.
   Code from this repo also uses `std::meta::parameters_of`, so add
   `-fparameter-reflection` to the compiler options.
3. Paste in your code. The library headers are not available on Compiler
   Explorer, so paste the headers you need (for example `include/ad/autograd.h`)
   into the source, above your own code.
4. To run the program as well as compile it, enable "Execute the code" in the
   compiler pane's Output menu (or add an execution pane).

If you use Boost, pick Boost 1.84 or older from the Libraries menu for now;
newer Boost versions are not yet packaged for this compiler.

### The gcc fork

The gcc fork
([yesmanchyk/gcc-mirror, branch `autograd-capable`](https://github.com/yesmanchyk/gcc-mirror/tree/autograd-capable))
is also on Compiler Explorer, as **x86-64 gcc (expr-reflect)**. It is in the
x86-64 gcc list (type `expr-reflect` in the compiler picker to find it) and is
rebuilt nightly from the tip of the `autograd-capable` branch. Its default
options are `-std=c++26 -freflection`.

[This example](https://godbolt.org/z/8hroz6cnx) reflects the body of a small function and rebuilds its
return expression as text at compile time. The same source compiles and runs
unchanged on both expr-reflect compilers, and fails on gcc trunk and on
"x86-64 clang (reflection - C++26)", which have no `body_of` or `operands_of`.

## Layout

The libraries are header-only, under `include/`:

- `include/ad/` — the AD engine (`autograd.h`, forward and reverse derivatives,
  higher-order Taylor AD, `is_invertible.hpp`).
- `include/ad/clang_only/` — the parts of the engine that need the clang
  (`expr-reflect`) fork: tensors (`tensor.h`, `autograd_tensor.h`) and the
  analyses (`is_continuous.hpp`, `limit_algebra.hpp`,
  `discontinuity_analysis.hpp`, `absolute_normal_form.hpp`).
- `include/cx_std/` — `constexpr` maths (`exp`, `log`, `sqrt`, `erfc`, `sin`,
  `cos`) for use in constant evaluation.

Include them by their path under `include/`, e.g. `#include <ad/autograd.h>`.
In CMake, link `reflection_ad::ad` (or `reflection_ad::cx_std`), which also
brings the compiler's flag profile.

Tests live under `tests/`, one directory per kind; [tests/README.md](tests/README.md)
says where a new test goes. Benchmarks live under `benchmarks/` (clang-only ones
in `benchmarks/clang_only/`), e.g. `ad_bench.cpp` and `ad_tensor_bench.cpp`. They
are built by CMake but not registered with CTest. `tools/` holds the Python
scripts that generate reference values and coefficient tables.

The two forks expose different reflection APIs, so tests are compiler-specific
for now. Each compiler has a **flag profile** (its reflection/stdlib flags), so
once a gcc engine exposes the same `ad::` interface its drivers can move to
`tests/common/` and run under both compilers unchanged. In CMake the profile is
the `reflection_ad::flags` target in [cmake/reflect_flags.cmake](cmake/reflect_flags.cmake),
which the library targets carry, so every test and benchmark gets it from there.

## Test Runner

`run_tests.py` is the one command for building and testing locally. It is a
thin driver over the [CMake presets](#building-with-cmake-presets); CI calls
the presets directly.
For each selected compiler (`--compiler clang`, `gcc` or `both`, the default)
and build type (`--build-type release`, the default, `debug`, `relwithdebinfo`,
or `all` for each in turn) it configures the preset (`clang-release`, ...) with
that compiler, builds, and runs `ctest`, then prints a summary. Everything else
(the flags, which tests exist, the `.fail.txt` comparison for `static_fail/`
and `known_gap/` tests) is defined in CMake, so `ctest` on its own gives the
same results. Build trees go in `build/cmake/<preset>`, shared with
`cmake --preset` and VS Code; `-B DIR` puts them in `DIR/<preset>` instead,
and `--clean` deletes them first. Arguments after `--` go to the CMake
configure step.

Tests under `tests/static_fail/` are expected **not** to compile (e.g. they
trip a `static_assert`); they pass only when the compiler's errors match the
`.fail.txt` baseline next to each test (see
[test_simple/README.md](test_simple/README.md)).
Tests under `tests/known_gap/{clang,gcc}/` record a known bug or missing
feature in that compiler's reflection fork: code it should accept but rejects.
They run on that compiler against their `.fail.txt`, so a test fails once the
gap is fixed (the file compiles); then move it into the normal test
directories. A `// TEST-PASSES-ON: clang` (or `gcc`) line makes such a test run
as an ordinary test on that other compiler.

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
`git status`. Host prerequisites: `git`, `cmake`, `ninja`, host
`clang`/`clang++`, `ld.lld`.

The recipes live in `tools/toolchain/`, and CI's toolchain cache runs the same
ones (`.github/scripts/build-*-toolchain.sh`), so a local build matches CI's.
CI builds clang itself as Release rather than RelWithDebInfo and strips the
result.

```bash
# one-time: build clang + runtimes into build/clang-p2996 (slow — builds LLVM)
python3 run_tests.py --compiler clang --build-compilers

# build and run every test (Release; the clang root defaults to build/clang-p2996)
python3 run_tests.py --compiler clang

# every build type, both compilers: the same presets CI runs
python3 run_tests.py --build-type all

# a single test, by ctest name or by path under tests/
python3 run_tests.py --compiler clang --tests ad_scalar_demo
python3 run_tests.py --compiler clang --tests 'common/ad_scalar_demo.cpp'

# build one test in Debug and start it under a debugger
python3 run_tests.py --compiler clang --tests ad_scalar_demo --debugger lldb

# every test under AddressSanitizer and UBSan (a Debug build, the
# <compiler>-sanitize preset); clang needs compiler-rt, which --build-compilers builds
python3 run_tests.py --sanitize

# build and run a benchmark (ITERATIONS=... tunes the workload)
ITERATIONS=1000000 python3 run_tests.py --compiler clang --benchmarks black_scholes_timing --run-executables

# use a clang built elsewhere (or set CLANG_P2996_ROOT), or any compiler binary
python3 run_tests.py --compiler clang --clang-root /path/to/clang-p2996
python3 run_tests.py --compiler gcc --gcc-executable /path/to/g++

# build clang from another clang-p2996 checkout (or set REFLECT_CLANG_SOURCE_DIR),
# into its own clang root so the default build/clang-p2996 is left untouched
python3 run_tests.py --compiler clang --build-compilers \
    --clang-source-dir ../clang-p2996 \
    --clang-root build/clang-other --clang-runtimes-dir build/libcxx-other
```

A `clang_only/` test asked for by name under `--compiler both` runs on clang
and is skipped, with a note, on GCC. `--verbose` shows the full build and
test commands.

Each demo prints its own `ALL CHECKS PASSED` / `ALL MATCH` on success.

## Building with CMake presets

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

## Formatting and linting

[pre-commit](https://pre-commit.com) checks formatting (clang-format 21 for C++, ruff for Python), lint and whitespace. CI runs the same hooks. Python tooling is managed by [uv](https://docs.astral.sh/uv/):

```sh
uv run pre-commit run --all-files   # check and fix everything
uv run pre-commit install           # optional: run the hooks on every commit
```

The Python helper scripts' dependencies are in the `scripts` group, e.g. `uv run --group scripts tools/black_scholes_sympy_value_check.py`.

## How to Debug with VSCode

The repo checks in a VS Code setup in `.vscode/`: CMake Tools in presets mode,
debug configurations, and IntelliSense from `build/compile_commands.json`.

1. Build the compiler(s) (see [Test Runner](#test-runner)); the presets look for
   them in `build/`.
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

The binaries carry an rpath to their compiler's runtime library, so they need
no `LD_LIBRARY_PATH`; on the command line, `lldb build/cmake/clang-debug/tests/ad_scalar_demo`
(or `gdb ...`) works as is.

The `(cppdbg)` configurations run gdb on Linux. On macOS, which has no gdb on
arm64, they run cpptools' `lldb-mi` with Apple's LLDB. Use them for GCC builds on
macOS: CodeLLDB's bundled lldb crashes showing a libstdc++ `std::unique_ptr` to a
class type. To make **Debug Test** use them, set `cmake.ctest.debugLaunchTarget`
to `Debug CTest test (cppdbg)`. To keep personal edits like that to the shared
`.vscode/` files out of `git status`, run `tools/vscode-local-edits.sh`
(`--undo` reverses it). The script's header says what to do when a pull changes
one of those files.

## Timing Benchmarks

To perform timing benchmarks, build a Release preset and run a benchmark setting iterations, for example:

```bash
cmake --preset clang-release && cmake --build --preset clang-release
ITERATIONS=1000000 ./build/cmake/clang-release/benchmarks/black_scholes_timing
```
