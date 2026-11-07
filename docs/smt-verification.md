# Verifying functions with an SMT solver

`tests/gcc_only/smtlib_verification.cpp` and
`tests/gcc_only/smtlib_loops.cpp` turn C++ functions into SMT-LIB, using
reflection on their bodies, and write scripts that a solver such as
[Z3](https://github.com/Z3Prover/z3) can check. This page says what the
translation covers, how to run it, and how to read the results. It was run
on 2026-11-07 with the GCC fork from `CppReflectionAD/gcc` `p4395`
(`3b65861`, plus the `conditional-statements` commits in `gcc-mirror`) and
Z3 4.16.0.

## What gets translated

`smt::define_fun(^^f)` (`tests/fixtures/smt/smtlib.hpp`) gives an SMT-LIB
`define-fun` that means the same as `f`, for an `f` over `int`:

| C++ | SMT-LIB |
|---|---|
| `int` | `(_ BitVec 32)`, so arithmetic wraps as on the hardware |
| `+ - *`, unary `-` | `bvadd bvsub bvmul`, `bvneg` |
| `< <= > >= == !=` | `bvslt bvsle bvsgt bvsge`, `=`, `(not (= …))` |
| `&& \|\| !` | `and or not` |
| `c ? a : b` | `(ite c a b)` |
| declaration, `x = e`, `x += e`, `++x` … | a `let` binding a fresh name, such as `m!3` |
| `if` / `else` | both branches, then an `ite` on the condition for each variable that differs |
| `return e` | `ret!k`, guarded by `returned!k`: the condition that an earlier return already happened |
| `while`, `do`/`while`, `for` | unrolled a fixed number of times (see [Loops](#loops)) |
| `break`, `continue` | `broken!k`, `jump!k`: where the rest of the body is skipped |

Anything else, such as a call or a division, makes `define_fun` throw, so
the test fails to compile rather than produce a formula with a different
meaning.

Two quirks of the GCC fork matter here:

- GCC folds `x < 0 ? -x : x` into its `abs` built-in before reflection sees
  it, and `expression_operator_of` has no operator for that. The test writes
  `abs_naive` with an `if` instead.
- An `if` without an `else` reflects with a `void` conversion as its else
  branch, which the translation treats as doing nothing.

## Loops

A solver can't reason about an unbounded number of iterations in one
formula, so `smt::define_fun(^^f, k)` unrolls each loop `k` times: each
iteration runs the body where the loop is still running (no return, no
break) and the condition holds, and merges the state with an `ite`, like an
`if`. It also defines `f_unwound`, which is true when no path that reaches
a loop would start iteration `k + 1`. A script then asks for an input, within
a precondition that bounds the loop, where `f_unwound` or the claim is
false. `unsat` proves the claim for every input the precondition allows;
the unrolling never silently cuts a run short, because then `f_unwound`
would be false. This is bounded model checking with an unwinding check.

GCC lowers a loop before reflection sees it
([gcc-loop-lowering.md](gcc-loop-lowering.md)):
`[goto C, B:, body…, C:, if (c) goto B; else goto X;, X:]`, without the first
goto for a `do`/`while`, after the init-statement for a `for`. The translator
finds that pattern with `is_goto_statement`, `is_label_statement` and
`label_of`. Within the body, a goto to `X` is a break; a goto to a later
label is a continue, and skips the statements up to that label (in a `for`,
the label before the increment).

Those three metafunctions are only in the `conditional-statements` commits
of `gcc-mirror`, not yet in the GCC pin. `smt::loops_supported` checks for
them, through argument-dependent lookup on `std::meta::info`; where they're
missing, `smtlib_loops` only prints that it skipped. On clang, which keeps
its loops, the translator doesn't handle loops yet.

Proving termination, or a claim for every `n` rather than a bounded range,
needs an inductive invariant instead of unrolling: written by the user, or
found by the solver from Horn clauses (Z3's Spacer). That isn't done yet.

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

`smtlib_loops` unrolls 16 times, and bounds `n` with a precondition:

| Script | Claim | Expected |
|---|---|---|
| `sum_to.smt2` | `sum_to(n) == n(n+1)/2` (a `while`), for `0 <= n <= 15` | `unsat` |
| `sum_to_for.smt2` | the same, written with a `for` | `unsat` |
| `sum_to_off_by_one.smt2` | the same, for a loop that stops at `n - 1` | `sat`: `n = 14` gives 91, not 105 |
| `sum_to_bound.smt2` | `sum_to`'s claim for `0 <= n <= 20` | `sat`: `n = 17`, where `sum_to_unwound` is false |
| `sum_capped.smt2` | a `break`: the sum stops at 21 once it reaches 20 | `unsat` |
| `sum_skip3.smt2` | a `continue`: the sum leaves out 3 | `unsat` |
| `countdown_do.smt2` | a `do`/`while`, which runs once even for `n <= 0`, for `-10 <= n <= 15` | `unsat` |

## Running it

The SMT-LIB is generated at compile time, so you need the GCC fork; the
solver can run anywhere. On a Mac, GCC runs in the Linux container (see
[pinned-gcc-container.md](pinned-gcc-container.md)) and Z3 on the host.

1. Build and run the test. With a directory argument it also writes the
   scripts there; it always prints them.

   ```sh
   docker exec -w /src -e LC_ALL=C <container> \
     python3 run_tests.py --compiler gcc \
       --gcc-executable /src/build/gcc-mirror/artifacts/bin/g++ \
       --tests smtlib_verification smtlib_loops
   docker exec -w /src <container> sh -c 'mkdir -p build/smt &&
     build/cmake/gcc-release/tests/smtlib_verification build/smt &&
     build/cmake/gcc-release/tests/smtlib_loops build/smt'
   ```

   `--gcc-executable` picks the GCC built from `gcc-mirror`, which the loop
   scripts need; leave it out to use the pinned one, which only produces the
   loop-free scripts.

   The tests also check the translation itself: `static_assert`s compare
   the generated `define-fun`s of `max_branchy` and `max_branchless` with
   the expected text, and check that the loops' definitions have
   `_unwound`, `broken!` and `jump!` parts.

2. Install Z3 (`brew install z3`, `apt install z3`, or `pip install
   z3-solver`) and check each script:

   ```sh
   for f in build/smt/*.smt2; do
     printf '%s: ' "$(basename $f)"; z3 "$f" | tr '\n' ' '; echo
   done
   ```

   The output on 2026-11-07, each in under 0.1 s:

   ```
   abs_nonnegative.smt2: sat ((x #x80000000))
   countdown_do.smt2: unsat
   max_equivalence.smt2: unsat
   median3.smt2: unsat
   sum_capped.smt2: unsat
   sum_skip3.smt2: unsat
   sum_to.smt2: unsat
   sum_to_bound.smt2: sat ((n #x00000011)  ((sum_to_unwound n) false))
   sum_to_for.smt2: unsat
   sum_to_off_by_one.smt2: sat ((n #x0000000e)  ((sum_to_off_by_one n) #x0000005b))
   ```

3. To see that a proof isn't vacuous, break the function and check that the
   solver notices. Making `median3`'s first return give `a` instead of `b`
   turns its result into `sat`:

   ```sh
   sed 's/(let ((ret!1 b))/(let ((ret!1 a))/' build/smt/median3.smt2 | z3 -in
   ```

   Likewise, claiming that `sum_capped` stops at 22 rather than 21 gives
   `sat`:

   ```sh
   sed 's/#x00000015 (tri n)/#x00000016 (tri n)/' build/smt/sum_capped.smt2 | z3 -in
   ```

## Adding a function

Write a function over `int` in a test, get its definition with
`std::define_static_string(smt::define_fun(^^f))` (or `(^^f, k)` if it has
loops), and add a script: declare its arguments with `declare-const`,
`assert` the precondition and the negation of the claim (and, with loops,
`(f_unwound …)` inside it), and end with `(check-sat)` (and `(get-value
(…))` to see a counterexample). In a test with loops, keep the call to
`define_fun` dependent on a template parameter, as `smtlib_loops` does, so
that a compiler without `label_of` skips it instead of failing.
