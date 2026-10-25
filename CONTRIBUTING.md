# Contributing

Everything here runs the same way locally and in CI. `run_tests.py` is the one
command for building the compilers and building and testing the repo; it drives
the [CMake presets](#cmake-presets), which CI calls directly. Where a new test
goes is in [tests/README.md](tests/README.md).

- [Prerequisites](#prerequisites)
- [Getting the compilers](#getting-the-compilers)
- [Building the compilers](#building-the-compilers)
- [Building and testing](#building-and-testing)
- [CMake presets](#cmake-presets)
- [Using it from another project](#using-it-from-another-project)
- [Debugging](#debugging)
- [Benchmarks](#benchmarks)
- [Formatting and linting](#formatting-and-linting)
- [CI checks](#ci-checks)
- [Merging a pull request](#merging-a-pull-request)
- [Reporting a compiler gap](#reporting-a-compiler-gap)
- [Bumping a compiler pin](#bumping-a-compiler-pin)

## Prerequisites

The tests need the two reflection compiler forks, which CI builds from the
repo's submodules. Download those builds, or build the compilers yourself (see
[Getting the compilers](#getting-the-compilers)). Released compilers can't
build this code.

- **Linux** (CI uses Ubuntu 24.04; WSL2 works the same): `git`, `cmake` (3.25
  or newer) and `ninja`. To build the compilers rather than download them, also
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

## Getting the compilers

CI builds both compilers from the submodule pins on Linux x86-64, Linux arm64
and macOS arm64, and publishes each build in the
[`toolchains` release](https://github.com/CppReflectionAD/cpp_reflection_ad/releases/tag/toolchains).
`--fetch-compilers` downloads the builds that match your checkout and machine,
checks them against their checksums, and unpacks them where
`--build-compilers` would put them, so the presets and `run_tests.py` find
them. It takes minutes rather than hours:

```bash
# both compilers, then every test with each
python3 run_tests.py --fetch-compilers

# just GCC, then build the tests without running them
python3 run_tests.py --compiler gcc --fetch-compilers --build-only
```

It's the same build the CI tests use: the asset is named after the pin and a
hash of the recipe, the same inputs as CI's toolchain cache key. Fetching again
after a pin bump replaces a fetched compiler, and does nothing if it's already
current. It never replaces a compiler you built, and `--build-compilers` won't
build over a fetched one: delete the directory first, or use another one
(`--clang-root`, `--gcc-build-dir`).

There's no asset for a pin or recipe that hasn't reached `main` yet, such as a
pin-bump branch, so build the compilers there. The Linux builds come from
Ubuntu 24.04 runners and may not run on a distribution with an older glibc. On
macOS, GCC finds the system
headers through the SDK it was built with, so a fetch warns when your Xcode's
SDK or linker differs from the CI runner's; if the fetched GCC then fails to
find headers or to link, build it instead.

## Building the compilers

The repo is **self-contained**: `--build-compilers` builds clang from its own
`clang-p2996` submodule, the `p4395` branch of
[CppReflectionAD/clang-p4395](https://github.com/CppReflectionAD/clang-p4395),
into `build/` — no externally pre-built compiler needed.
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

GCC comes from the `gcc-mirror` submodule, the `p4395` branch of
[CppReflectionAD/gcc](https://github.com/CppReflectionAD/gcc), into
`build/gcc-p2996`. On macOS it is built from `gcc-darwin-reflect` instead: the
`p4395` branch of
[CppReflectionAD/gcc-darwin-arm64](https://github.com/CppReflectionAD/gcc-darwin-arm64),
our fork of iains' Darwin port. That branch is an iains snapshot with
`gcc-mirror`'s reflection commits cherry-picked on top (see
[Bumping a compiler pin](#bumping-a-compiler-pin)).

Clones made before the submodule moved to the fork still fetch it from iains,
and may hold commits from the old replay. Update them once, after pulling:

```bash
git submodule sync gcc-darwin-reflect && git submodule update gcc-darwin-reflect
```

Likewise, `gcc-mirror` moved from `yesmanchyk/gcc-mirror` (branch
`autograd-capable`) to `CppReflectionAD/gcc` (branch `p4395`), and clones made
before the move still fetch it from the old repository. Update them once,
after pulling:

```bash
git submodule sync gcc-mirror && git submodule update gcc-mirror
```

And `clang-p2996` moved from `siqiyyyy/clang-p2996` (branch `expr-reflect`) to
`CppReflectionAD/clang-p4395` (branch `p4395`), with the same pin. Update
clones made before the move once, after pulling:

```bash
git submodule sync clang-p2996 && git submodule update clang-p2996
```

To work on GCC on a Mac, branch from `p4395` in `gcc-darwin-reflect`:
`--build-compilers` builds whatever is checked out there. Then move the
commits to `gcc-mirror` and open the pull request against `p4395` in
CppReflectionAD/gcc, which Linux CI tests:

```bash
git -C gcc-mirror fetch ../gcc-darwin-reflect <branch>
git -C gcc-mirror cherry-pick <commits>
```

Once `gcc-mirror` has them, they reach `gcc-darwin-reflect`'s `p4395` with the
next pin bump.

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
# <compiler>-sanitize preset); clang needs compiler-rt, which fetched and built clangs have
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

## Using it from another project

`cmake --install build/cmake/<preset> --prefix <dir>` installs the headers and
a CMake package. Another project then uses it with:

```cmake
project(my_project CXX)
find_package(reflection_ad 0.1 CONFIG REQUIRED)   # -DCMAKE_PREFIX_PATH=<dir>
target_link_libraries(my_target PRIVATE reflection_ad::ad)
```

`reflection_ad::ad` is the AD engine and `reflection_ad::cx_std` the constexpr
maths. The project must be built with one of the reflection forks, and needs
CMake 3.25 or later. The package stores no compiler paths: `find_package` works
out the flag profile (the reflection flags and the fork's standard library) for
the project's own compiler, and checks it can build reflection code. Until
1.0, a minor version may break the API, so asking for 0.1 accepts any 0.1.x
and nothing else; the version is in `project()` in `CMakeLists.txt`. The
`install_consumer` test (`tests/install/`) installs each build and builds a
small project against it.

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

1. Fetch or build the compiler(s) (see
   [Getting the compilers](#getting-the-compilers)); the presets look for them
   in `build/`.
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

Two of the hooks check generated values the tests rely on. Each runs its
generator with `--check`, which recomputes the values in high precision and
fails if the committed copy differs:

- `tools/black_scholes_sympy_value_check.py` writes
  `tests/fixtures/mc_sim/black_scholes_reference.hpp`, the exact Black-Scholes
  price and first derivatives that `black_scholes_demo` checks against. Run it
  without `--check` to rewrite the header after changing its inputs.
- `tools/gen_erfc_chebyshev.py` prints the Chebyshev table in
  `include/cx_std/cx_erfc.hpp`; paste its output over the table.

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
warm. `GCC pins` and `Submodule pins` run nightly too, in case a pin stops
being fetchable. `Submodule pins` checks that each submodule's pin is on the
branch its `.gitmodules` entry names, through GitHub's compare API. Pull
requests test the merge with `main`. `pre-commit`, `Submodule pins` and the two
test workflows also run in the merge queue (`merge_group`), on each queued pull
request combined with `main` and the pull requests queued ahead of it. A queue run
can't read a toolchain cache saved by its pull request, so a pull request that
changes a toolchain rebuilds it in the queue too.

| Check | Workflow | Required | Reproduce locally |
|---|---|---|---|
| `pre-commit` | `pre-commit.yml` | yes | `uv lock --check && uv run pre-commit run --all-files` |
| `Clang tests` | `clang_prebuilt_cache.yml` | yes | `python3 run_tests.py --compiler clang --build-type all && python3 run_tests.py --compiler clang --sanitize` |
| `GCC tests` | `gcc_prebuilt_cache.yml` | yes | `python3 run_tests.py --compiler gcc --build-type all && python3 run_tests.py --compiler gcc --sanitize` |
| `sanitize (ASan+UBSan)` | both test workflows | yes, through `Clang tests` and `GCC tests` | `python3 run_tests.py --sanitize` |
| `Submodule pins` | `submodule-pins.yml` | yes | `.github/scripts/check-submodule-pins.sh` (needs `gh` logged in) |
| `test (Release, x86-64-v3)` | both test workflows | not yet | `cmake --preset clang-x86-64-v3 && cmake --build --preset clang-x86-64-v3 && ctest --preset clang-x86-64-v3` (or `gcc-`) |
| `test (<build type>, macOS arm64)` | both test workflows | not yet ([#94](https://github.com/CppReflectionAD/cpp_reflection_ad/issues/94) C9) | `python3 run_tests.py --build-type all` and `python3 run_tests.py --sanitize`, on a Mac |
| `test (<build type>, Linux arm64)` | both test workflows | not yet ([#94](https://github.com/CppReflectionAD/cpp_reflection_ad/issues/94) C9) | `python3 run_tests.py --build-type all` and `python3 run_tests.py --sanitize`, on an arm64 Linux machine |
| `GCC pins` | `gcc-pins.yml` | not yet | `.github/scripts/check-gcc-pins.sh` |

`Clang tests` and `GCC tests` are summary jobs. Each passes only when its
workflow's `toolchain` job, its three `test (<build type>)` legs (one per build
type, so they match `--build-type all`) and its `sanitize (ASan+UBSan)` job
pass.

The `toolchain` job restores the compiler from the Actions cache, or builds it
on a miss: about 1.5 hours for clang, about 30 minutes for GCC. The cache key
is the submodule's pinned commit plus a hash of
`.github/scripts/build-<compiler>-toolchain.sh`, `tools/toolchain/common.py`
and `tools/toolchain/<compiler>.py`. So a change to one of those files, or a
pin bump, rebuilds the toolchain, and a change to `common.py` rebuilds both.
A cache saved by a pull request can't be read from `main`, so the toolchain is
built again after merging. Workflow YAML isn't in the key.

After building GCC, the `toolchain` job also runs the fork's own reflection
tests (`gcc/testsuite/g++.dg/reflect`) in the build tree, and fails before
saving the cache if any of them fails, so a pin that breaks them is never
cached. Their `g++.sum` and `g++.log` are uploaded as the job's artifact. To
run them on the current pin without a pin bump, start `Build GCC Toolchain
(Cache)` by hand (`workflow_dispatch`) with `force_build` set: it builds and
tests GCC on the chosen runner even on a cache hit, and doesn't save it.

The clang workflow also builds and caches clang on macOS arm64 in its
`toolchain-macos` job, under the same key, which records the runner's OS and
architecture. The macOS prefix isn't stripped, since the recipe's strip step
uses GNU strip options.

The GCC workflow also builds and caches GCC on macOS arm64, from
`gcc-darwin-reflect`, in its `toolchain-macos` job. Its key uses that
submodule's commit and also records the runner's SDK (its path, which GCC is
configured with, and its build version) and linker version, from
`.github/scripts/macos-sdk-id.sh`. So a runner image with another Xcode, SDK
or linker rebuilds it. The macOS test legs fail early if their runner's SDK or
linker differs from the toolchain job's, which can happen while GitHub rolls
out a new runner image; wait for the rollout to finish, then re-run all jobs.

Both workflows also build and cache their compiler on Linux arm64
(`ubuntu-24.04-arm`) in a `toolchain-arm64` job, with the same recipe and key
as on x86-64. The key records the runner's architecture, so the two don't
share an entry.

After each toolchain job, an `archive` job (`toolchain-archive.yml`) checks
whether the `toolchains` release already has that toolchain's asset for
`--fetch-compilers`. If it doesn't, the job packs the cached toolchain,
deletes the original, fetches the archive into `build/` and runs the Release
preset and the toolchain check with it, so the archive is known to work once
moved. On `main`, a `publish` job then uploads it; it's the only job with
write access to the repository. So a pin bump's pull request checks its
archives, and its merge publishes them. The `archive` jobs aren't required
checks and don't run in the merge queue.

## Merging a pull request

`main` changes only through pull requests, and they are squash-merged through
a merge queue. A pull request can join the queue once it has:

- `pre-commit`, `Submodule pins`, `Clang tests` and `GCC tests` passing (see
  [CI checks](#ci-checks)). So a pin bump can merge only after its pins are
  merged into their fork branches;
- an approving review from someone other than whoever pushed last. A push
  dismisses earlier approvals;
- every review thread resolved.

Then click "Merge when ready". GitHub runs the required checks again on the
pull request combined with `main` and the pull requests queued ahead of it,
and merges it if they pass. If they fail, it leaves the queue and stays open.
The branch doesn't need updating from `main`, so an approval isn't lost to an
update push. A queue run that has
to rebuild a toolchain can take a couple of hours; the queue waits up to 4
hours for the checks.

Repository admins can dismiss reviews, and can bypass these rules, but only
through a pull request: nobody pushes to `main` directly. The rules come from
the repository's "main" ruleset on the default branch (Settings → Rules →
Rulesets), which
[#94](https://github.com/CppReflectionAD/cpp_reflection_ad/issues/94) C5
tracks. It applies only to pull requests into `main`, so one stacked on
another pull request's branch shows as mergeable without a review until it is
retargeted to `main`.

## Reporting a compiler gap

A bug or missing feature in one of the reflection forks is a compiler gap.
The forks ([CppReflectionAD/clang-p4395](https://github.com/CppReflectionAD/clang-p4395),
[CppReflectionAD/gcc](https://github.com/CppReflectionAD/gcc) and
[CppReflectionAD/gcc-darwin-arm64](https://github.com/CppReflectionAD/gcc-darwin-arm64))
have issues turned off, and their READMEs send bug reports here. Report one
as an issue in this repository with the `CompilerGap` label:

- Give a small reproducer that uses only `<meta>`, in the fork's own API, so a
  fix can add it to the fork's tests (for GCC, as a `g++.dg/reflect` test).
- Say which pin shows it, what the other compiler does, and what it blocks
  here.
- If code here runs into it, record it as a `known_gap/` test (see
  [tests/README.md](tests/README.md)) that names the issue.

A fix is a pull request into the fork's `p4395` branch. Then bump the pin
here, moving the `known_gap/` test in the same pull request (see
[Bumping a compiler pin](#bumping-a-compiler-pin)), and close the issue.

## Bumping a compiler pin

The compilers are the submodules `clang-p2996` (the `p4395` branch of
`CppReflectionAD/clang-p4395`), `gcc-mirror` (the `p4395` branch of
`CppReflectionAD/gcc`) and, on macOS only, `gcc-darwin-reflect`. A bump is
its own pull request, along with the test changes the new pin forces, such as
moving a `known_gap/` test it fixes (see [tests/README.md](tests/README.md)).
For `clang-p2996` or `gcc-mirror`:

```bash
git -C clang-p2996 fetch origin p4395
git -C clang-p2996 checkout --detach <commit>
git add clang-p2996

# rebuild the compiler from the new pin and run every test with it
python3 run_tests.py --compiler clang --build-compilers --build-type all
```

In the pull request, say what the bump brings in and compare the test results
of the old and new pins. Change `.gitmodules` only if the URL or branch
changes, and then run `git submodule sync` too.

Each fork pins from its `p4395` branch, which only moves forward, through pull
requests merged with a merge commit. A rule on the branch enforces this in all
three forks: no deletion or force push, and a pull request (no approval
needed), merged with "Create a merge commit". Squash and rebase merging are
also turned off in each fork's settings, so every merge button there offers
only a merge commit. A squash or rebase merge would give the
pinned commit a new SHA and leave it on no branch, and a force push would drop
it; either way, older commits of this repository could no longer check out
their compiler.

CI rebuilds the toolchain on the pull request, and on `main` again after
merging (see [CI checks](#ci-checks)). Open pull requests rebuild it as well
once they test against the new `main`.

`gcc-darwin-reflect` follows `gcc-mirror`, so when the `gcc-mirror` pin moves,
bring its new commits over to `p4395` in the same pull request. `git cherry`
matches commits by content, so this works from any clone, skips `gcc-mirror`'s
merge commits, and picks only what `p4395` lacks:

```bash
new=$(git rev-parse :gcc-mirror)
cd gcc-darwin-reflect
git fetch origin p4395
git fetch --no-tags ../gcc-mirror "$new"
git checkout --detach origin/p4395
git cherry HEAD "$new" "$(git merge-base HEAD "$new")" | sed -n 's/^+ //p' |
  git cherry-pick -x --stdin
git push origin HEAD:refs/heads/<pr-branch>
cd ..
git add gcc-darwin-reflect
```

Then open a pull request from `<pr-branch>` into `p4395` in
CppReflectionAD/gcc-darwin-arm64, and merge it with a merge commit before this
repository's pull request merges. The pin can stay at `<pr-branch>`'s tip,
since the merge commit keeps it on `p4395`.

Run the `git cherry` line again afterwards: it should print only `-` lines,
meaning every reflection commit in `gcc-mirror` is in `p4395`. If it already
does before you start, `git cherry-pick` stops with "empty commit set passed":
there is nothing to bring over. `.github/scripts/check-gcc-pins.sh` runs the
same check on the staged pins, in a scratch clone, and the `GCC pins` CI check
runs it on every pull request.

The iains snapshot under `p4395` must have the same upstream GCC base as
`gcc-mirror`. If `gcc-mirror` merged a newer upstream GCC, the `git cherry`
list above also holds upstream commits; move to a newer iains snapshot instead.
The fork has only the branches someone pushed to it, so push the snapshot's
branch from iains first, then rebuild the stack on it:

```bash
cd gcc-darwin-reflect
git fetch https://github.com/iains/gcc-darwin-arm64.git <branch>
git push origin FETCH_HEAD:refs/heads/<branch>
git checkout --detach FETCH_HEAD
```

Then run the `git cherry` pipeline above (from `git cherry` on). Record the
old `p4395` as a parent, so its pins stay on the branch, while keeping the tree
of the rebuilt stack, and push and merge as above:

```bash
git merge -s ours --no-edit origin/p4395
git push origin HEAD:refs/heads/<pr-branch>
```

The `-s ours` merge takes nothing from the old stack, so the `git cherry` line
still prints only `-` lines.
