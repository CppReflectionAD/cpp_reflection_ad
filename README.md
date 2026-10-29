# cpp_reflection_ad

> [!WARNING]
> **`.vscode/` is now tracked ([#124](https://github.com/CppReflectionAD/cpp_reflection_ad/pull/124)).** Back up your local `.vscode/settings.json`,
> `launch.json` and `extensions.json` before you pull: git replaces ignored files
> with tracked ones without asking. Afterwards, copy your changes back and run
> `tools/vscode-local-edits.sh` so git leaves them alone. Details in [#125](https://github.com/CppReflectionAD/cpp_reflection_ad/issues/125).

Proof of concept for automatic differentiation of C++ code using static reflection. Based on (and hoping to extend) the C++ reflection proposal P2996R13:
https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r13.html

## Try it on Compiler Explorer (godbolt)

The P4395 clang fork this repo builds on
([CppReflectionAD/clang-p4395, branch `p4395`](https://github.com/CppReflectionAD/clang-p4395/tree/p4395))
is available on [Compiler Explorer](https://godbolt.org/) as
**x86-64 clang (expr-reflect)**, so you can try expression reflection and this
library in the browser without building a compiler. It is rebuilt nightly from
the tip of the `p4395` branch, the same way Bloomberg's P2996 fork is
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
([CppReflectionAD/gcc, branch `p4395`](https://github.com/CppReflectionAD/gcc/tree/p4395))
is also on Compiler Explorer, as **x86-64 gcc (expr-reflect)**. It is in the
x86-64 gcc list (type `expr-reflect` in the compiler picker to find it) and is
rebuilt nightly from the tip of the `p4395` branch. Its default
options are `-std=c++26 -freflection`.

[This example](https://godbolt.org/z/8hroz6cnx) reflects the body of a small function and rebuilds its
return expression as text at compile time. The same source compiles and runs
unchanged on both expr-reflect compilers, and fails on gcc trunk and on
"x86-64 clang (reflection - C++26)", which have no `body_of` or `operands_of`.

## Layout

The libraries are header-only, under `include/`:

- `include/ad/` — the AD engine (`autograd.h`, forward and reverse derivatives,
  higher-order Taylor AD, `is_invertible.hpp`), tensors (`tensor.h`,
  `autograd_tensor.h`) and the analyses `is_continuous.hpp`,
  `limit_algebra.hpp`, `discontinuity_analysis.hpp` and
  `absolute_normal_form.hpp`.
- `include/cx_std/` — `constexpr` maths (`exp`, `log`, `sqrt`, `erfc`, `sin`,
  `cos`) for use in constant evaluation.

Include them by their path under `include/`, e.g. `#include <ad/autograd.h>`.
In CMake, link `reflection_ad::ad` (or `reflection_ad::cx_std`), which also
brings the compiler's flag profile.

Tests live under `tests/`, one directory per kind; [tests/README.md](tests/README.md)
says where a new test goes. Benchmarks live under `benchmarks/` (clang-only ones
in `benchmarks/clang_only/`), e.g. `ad_bench.cpp` and `ad_tensor_bench.cpp`.
`tools/` holds the Python scripts that generate reference values and
coefficient tables.

The two forks expose different reflection APIs, so tests are compiler-specific
for now. Each compiler has a **flag profile** (its reflection/stdlib flags), so
once a gcc engine exposes the same `ad::` interface its drivers can move to
`tests/common/` and run under both compilers unchanged. In CMake the profile is
the `reflection_ad::flags` target in [cmake/reflect_flags.cmake](cmake/reflect_flags.cmake),
which the library targets carry, so every test and benchmark gets it from there.

## Building and testing

The tests build with the repo's two reflection compiler forks, which it builds
from its submodules. [CONTRIBUTING.md](CONTRIBUTING.md) says how, and covers
debugging, linting, CI and bumping a compiler pin.
