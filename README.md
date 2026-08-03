# cpp_reflection_ad

Proof of concept for automatic differentiation of C++ code using static reflection. Based on (and hoping to extend) the C++ reflection proposal P2996R13:
https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r13.html

## Test Runner

The top-level test harness is `run_tests.py`.

It can:
- select `clang`, `gcc`, or both,
- build the corresponding compiler submodules,
- compile all `.cpp` files under `tests/`,
- optionally run the produced executables,
- report compile-time and runtime failures separately.

If `--build-compilers` is not provided, the script assumes the selected compiler binaries have already been built. If the expected executable is missing, it stops with an error instead of building automatically.

Examples:

```bash
python3 run_tests.py --compiler clang
python3 run_tests.py --compiler clang --build-compilers
python3 run_tests.py --compiler both --build-compilers --run-executables
python3 run_tests.py --compiler gcc --tests hello_world.cpp --run-executables
```
