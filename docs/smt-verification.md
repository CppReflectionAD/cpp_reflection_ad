# Verifying functions with an SMT solver

`tests/gcc_only/smtlib_verification.cpp` turns C++ functions into SMT-LIB,
using reflection on their bodies, and writes scripts that a solver such as
[Z3](https://github.com/Z3Prover/z3) can check. This page says what the
translation covers, how to run it, and how to read the results. It was run
on 2026-11-07 with the GCC fork from `CppReflectionAD/gcc` `p4395`
(`3b65861`, plus the `conditional-statements` commits in `gcc-mirror`) and
Z3 4.16.0.

## What gets translated

`smt::define_fun(^^f)` (`tests/fixtures/smt/smtlib.hpp`) gives an SMT-LIB
`define-fun` that means the same as `f`, for a loop-free `f` over `int`:

| C++ | SMT-LIB |
|---|---|
| `int` | `(_ BitVec 32)`, so arithmetic wraps as on the hardware |
| `+ - *`, unary `-` | `bvadd bvsub bvmul`, `bvneg` |
| `< <= > >= == !=` | `bvslt bvsle bvsgt bvsge`, `=`, `(not (= …))` |
| `&& \|\| !` | `and or not` |
| `c ? a : b` | `(ite c a b)` |
| declaration, `x = e`, `x += e` … | a `let` binding a fresh name, such as `m!3` |
| `if` / `else` | both branches, then an `ite` on the condition for each variable that differs |
| `return e` | `ret!k`, guarded by `returned!k`: the condition that an earlier return already happened |

Anything else, such as a loop, a call or a division, makes `define_fun`
throw, so the test fails to compile rather than produce a formula with a
different meaning. Loops would need invariants, which a body doesn't carry.

Two quirks of the GCC fork matter here:

- GCC folds `x < 0 ? -x : x` into its `abs` built-in before reflection sees
  it, and `expression_operator_of` has no operator for that. The test writes
  `abs_naive` with an `if` instead.
- An `if` without an `else` reflects with a `void` conversion as its else
  branch, which the translation treats as doing nothing.

## The scripts

Each script defines one or two functions, states a claim about them, and
asserts its negation. `unsat` means no input breaks the claim: it is proved
for all 2³² (or 2⁶⁴, 2⁹⁶) inputs. `sat` means the solver found an input
that breaks it.

| Script | Claim | Expected |
|---|---|---|
| `median3.smt2` | `median3(a, b, c)` is one of its arguments, with at least two arguments `<=` it and at least two `>=` it | `unsat` |
| `max_equivalence.smt2` | `max_branchy` and `max_branchless` return the same value for every input | `unsat` |
| `abs_nonnegative.smt2` | `abs_naive(x) >= 0` | `sat`, with `x = #x80000000`: `-INT_MIN` wraps to `INT_MIN` |

The last one is there to show the solver catching a real bug, the
`INT_MIN` overflow, that tests on hand-picked inputs usually miss.

## Running it

The SMT-LIB is generated at compile time, so you need the GCC fork; the
solver can run anywhere. On a Mac, GCC runs in the Linux container (see
[pinned-gcc-container.md](pinned-gcc-container.md)) and Z3 on the host.

1. Build and run the test. With a directory argument it also writes the
   scripts there; it always prints them.

   ```sh
   docker exec -w /src -e LC_ALL=C <container> \
     python3 run_tests.py --compiler gcc --tests smtlib_verification
   docker exec -w /src <container> sh -c \
     'mkdir -p build/smt && build/cmake/gcc-release/tests/smtlib_verification build/smt'
   ```

   Add `--gcc-executable /src/build/gcc-mirror/artifacts/bin/g++` to use a
   GCC built from `gcc-mirror` rather than the pinned one.

   The test also checks the translation itself: `static_assert`s compare
   the generated `define-fun`s of `max_branchy`, `max_branchless` and
   `abs_naive` with the expected text.

2. Install Z3 (`brew install z3`, `apt install z3`, or `pip install
   z3-solver`) and check each script:

   ```sh
   for f in median3 max_equivalence abs_nonnegative; do
     printf '%s: ' $f; z3 build/smt/$f.smt2 | tr '\n' ' '; echo
   done
   ```

   The output on 2026-11-07:

   ```
   median3: unsat
   max_equivalence: unsat
   abs_nonnegative: sat ((x #x80000000))
   ```

3. To see that a proof isn't vacuous, break the function and check that the
   solver notices. Making `median3`'s first return give `a` instead of `b`
   turns its result into `sat`:

   ```sh
   sed 's/(let ((ret!1 b))/(let ((ret!1 a))/' build/smt/median3.smt2 | z3 -in
   ```

## Adding a function

Write a loop-free function over `int` in the test, get its definition with
`std::define_static_string(smt::define_fun(^^f))`, and add a script: declare
its arguments with `declare-const`, `assert` the negation of the claim, and
end with `(check-sat)` (and `(get-value (…))` to see a counterexample).
