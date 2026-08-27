# cpp_reflection_ad

Proof of concept for automatic differentiation of C++ code using static reflection. Based on (and hoping to extend) the C++ reflection proposal P2996R13:
https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r13.html

## Tests

Reflection-AD tests live under `tests/`:

- `tests/clang_only/` — the clang (`expr-reflect`) engine (`autograd.h`,
  `autograd_tensor.h`, `tensor.h`) and its tests: self-checking demos
  (`ad_scalar_demo.cpp`, `ad_tensor_demo.cpp`) and benchmarks (`ad_bench.cpp`,
  `ad_tensor_bench.cpp`, which carry `// TEST-FLAGS: -O2`).
- `tests/gcc_only/` — reserved for the gcc (`no_expression_kind`) port.
- `tests/` (directly) — compiler-agnostic tests run under **both** compilers.

The two forks expose different reflection APIs, so tests are compiler-specific
for now. Each compiler has a **flag profile** (its reflection/stdlib flags), so
once a gcc engine exposes the same `ad::` interface its drivers can move up to
`tests/` and run under both compilers unchanged.

## Test Runner

`run_tests.py` selects `clang`/`gcc`/both, compiles every `.cpp` under `tests/`
(respecting the `clang_only`/`gcc_only`/shared dirs), optionally runs the
binaries, and reports compile vs runtime failures. A test may add flags via a
`// TEST-FLAGS: ...` comment near its top (benchmarks use it for `-O2`).

The repo is **self-contained**: `--build-compilers` builds clang from its own
`clang-p2996` submodule into `build/` — no externally pre-built compiler needed.
It checks out the submodule, builds clang (host `clang` + `lld`), builds the
`libc++`/`libc++abi`/`libunwind` runtimes with that clang, and syncs the fork's
`<meta>` header. Host prerequisites: `git`, `cmake`, `ninja`, host
`clang`/`clang++`, `ld.lld`.

```bash
# one-time: build clang + runtimes into build/clang-p2996 (slow — builds LLVM)
python3 run_tests.py --compiler clang --build-compilers

# run the tests (clang root defaults to build/clang-p2996)
python3 run_tests.py --compiler clang --run-executables

# a single test; --verbose prints the exact compile commands
python3 run_tests.py --compiler clang --tests 'ad_scalar_demo.cpp' --run-executables

# use a clang built elsewhere (or set CLANG_P2996_ROOT)
python3 run_tests.py --compiler clang --clang-root /path/to/clang-p2996 --run-executables
```

Each demo prints its own `ALL CHECKS PASSED` / `ALL MATCH` on success.

## How to Debug with VSCode

1. Install these extensions (or verify they are already installed):
    - `vadimcn.vscode-lldb`
    - `ms-vscode.cmake-tools`
2. Copy or adapt the files from `copy_to_.vscode` into your workspace `.vscode` folder.
3. Build the compiler artifacts before selecting kits:
    - `python3 run_tests.py --compiler clang --build-compilers`
    - `python3 run_tests.py --compiler gcc --build-compilers`
4. In VS Code CMake Tools, select one of these kits:
    - `Clang reflection`
    - `GCC reflection (macOS)`
    - `GCC reflection (Linux)`

Use the kit that matches your host OS: macOS -> `GCC reflection (macOS)`, Linux -> `GCC reflection (Linux)`.

You should now be able to build and debug the tests in VSCode.

## Timing Benchmarks

To perform timing benchmarks, build with Release option and run a benchmark setting iterations, for example:

```bash
ITERATIONS=1000000 ./build/cmake/tests/black_scholes_timing
```
