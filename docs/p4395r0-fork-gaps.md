# P4395R0 vs. the compiler forks: implementation gaps

This compares what [P4395R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4395r0.pdf)
(2026-10-23, "Reflecting Function Bodies") proposes with what the two
compiler forks do today. It covers the §3 API and its semantic promises, and
the code shown in §2 and §5. Measured on 2026-11-04, using CI's builds of the
compilers `main` pins.

| Column | What it is |
|---|---|
| **clang** | `CppReflectionAD/clang-p4395` at `4efbc03`, `main`'s pin. The `p4395` tip, `d079753`, is the merge commit of that same tree. |
| **GCC** | `CppReflectionAD/gcc` at `458ffb3`, `main`'s pin and the `p4395` tip. |

Each probe in [`p4395r0-fork-gaps/`](p4395r0-fork-gaps/) is a small file
asserting one claim from the paper, spelled with `std::meta` as the paper
spells it. `ad::meta` is now the same namespace, so it isn't measured
separately.

✅ matches the paper · ⚠️ present but behaves differently · ❌ missing or wrong ·
ℹ️ the paper promises nothing here

## Headlines

Both forks now have all 24 proposed functions, and §5's `build()` compiles as
printed on both. What's left is about tree shape:

1. **The paper's own §3.5 `?:` example fails on clang.** `operands_of(e).size()`
   is 1, not 3. Clang wraps the conditional in an implicit lvalue-to-rvalue
   conversion. It does the same to every id-expression used as a value, so
   operands such as `x` in `x * 2` fail `is_id_expression` and `declaration_of`.
   GCC has conversion nodes too, but in a different place: it wraps a use of a
   reference and leaves other value uses bare (#249). The paper has no notion
   of conversion nodes.
2. **Neither fork extends `std::meta::operators` with `?:`** (§3.6). Both
   instead add `is_conditional_operator`, `condition_of`, `true_expression_of`
   and `false_expression_of`, and `is_operator_expression` is false on a `?:`.
3. **GCC lowers `+=`** to `= … + …`, so §2.5's `update` example never sees
   `op_plus_equals` (#246). `-=`, `*=` and `/=` go the same way.
4. **On GCC, a block isn't always a statement.** A plain nested block that
   declares nothing is flattened into its parent (#245). An `if constexpr` is
   folded away, and what's left of it follows its own rules: a two-statement
   taken branch with statements after it stays a compound statement, even if it
   declares nothing (#250).
5. **On GCC, a `for` or `while` satisfies `is_compound_statement`.** It is
   lowered, and `statements_of` on it returns the lowered pieces, one of which
   fails `is_statement` (#247).
6. **On GCC, a class-type argument passed by value is an opaque operand.** It
   satisfies `is_expression` but none of the paper's other predicates, and has
   no operands, so the walk §3.5 describes can't reach the argument (#248).
7. **A null statement is an expression statement on neither fork** (#237). The
   fix is in review; see [Open gaps](#open-gaps).

## API and semantics, claim by claim

| § | Paper API / claim | clang | GCC | Probe |
|---|---|---|---|---|
| 3.1 | `has_body` | ✅ | ✅ | p01 |
| 3.1 | `body_of` on a declared-only function | ℹ️ not a constant expression; can't be caught, because the fork has no `std::meta::exception` | ℹ️ throws `std::meta::exception` "function is not defined" | p02, p20 |
| 3.1 | `body_of` gives a compound statement | ✅ | ✅ | d01 |
| 3.2 | `statements_of` lists the block's statements in order | ✅ | ⚠️ a nested block that declares nothing is flattened into the parent list (#245); one that declares something stays a compound statement | p04, d01 #2 |
| 3.2 | `is_compound_statement` | ✅ | ⚠️ true on a `for` or `while` (#247); a nested block that declares nothing is never returned, so it can't be asked (#245) | p04, p24 |
| 3.2 | every `statements_of` element satisfies `is_statement` | ✅ | ⚠️ holds for a function body, but not inside a lowered loop, whose list includes an expression (#247) | p24, d01 #2 |
| 3.2 / 4.2 | the discarded `if constexpr` branch is not returned | ⚠️ its statements are gone, but `else_statement_of` is an empty compound, not null | ⚠️ the discarded branch is gone, but so is the `if`. With the `if` as the whole body, the taken branch's statements become the body's, so the paper's "only one statement" holds only if that branch has one; if no branch is taken there are none. With statements after it, a two-statement branch is one compound statement and a one-statement branch is flattened (#250) | p03, d01 #1 |
| 3.2 | `is_declaration_statement` | ✅ | ✅ | p06, d01 #4 |
| 3.2 | `is_expression_statement` | ✅ | ✅ (p04's failure here comes from #245 shifting the indices) | p13, d01 #2 |
| 3.2 | a null statement is an expression statement | ❌ a statement of no kind (#237) | ❌ left out of `statements_of` (#237) | p21 |
| 3.2 | `is_return_statement` | ✅ | ✅ | p12, d01 #2 |
| 3.3 | `is_simple_declaration`, excluding structured bindings | ✅ | ✅ | p06 |
| 3.3 / 5.1 | `declarators_of` / `initializers_of` (`double v = …, w = …;` gives 2 of each) | ✅ | ✅ | p05 |
| 3.4 | `return_operand_of`, including `return {x, y};` → the constructor call | ✅ | ✅ | p12 |
| 3.5 | `expression_of` | ✅ | ✅ | p13 |
| 3.5 | `is_expression`: statements aren't expressions | ✅ | ✅ | d01 |
| 3.5 | `is_operator_expression` (built-in or overloaded, including `&&`, `\|\|`, `!`) | ✅ | ✅ | p10, p11, p16 |
| 3.5 | `expression_operator_of` | ✅ | ⚠️ `p.x += e` is lowered to `p.x = p.x + e`, so `op_plus_equals` never appears (#246) | p10, p13, d01 #7 |
| 3.5 | `is_function_call`, also true for an overloaded operator | ✅ | ✅ | p08, p10 |
| 3.5 | `callee_of` is the raw callee expression; `is_id_expression` + `declaration_of` resolve it | ✅ | ✅ | p08 |
| 3.5 | `operands_of` on a call: the arguments, defaulted ones included | ✅ (p09's failure on clang is the `is_id_expression` row) | ✅ | p09 |
| 3.5 | operands are leaves or sub-expressions to recurse into | ✅ | ⚠️ the walk ends, and `build()` runs, but a class-type argument passed by value has no operands, so it never reaches the argument (#248) | p17, p23, d01 #6 |
| 3.5 | `is_literal` | ✅ | ✅ | p11 |
| 3.5 | `is_id_expression` | ⚠️ exists, but value uses arrive wrapped in an implicit conversion, so `operands_of(v + 1)[0]` is not an id-expression | ⚠️ a use of a reference arrives wrapped in a conversion (#249); other uses are bare | p07, p09, p22, d01 #8 |
| 3.5 | `declaration_of(id)` matches `declarators_of(decl)[i]` | ⚠️ only after stripping the implicit conversion (`while (is_conversion(e)) e = operands_of(e)[0];`); on the operand as returned it is not a constant expression | ✅ | p07, p19 |
| 3.5 | the `max` example: `operands_of(e).size() == 3` | ❌ size is 1 (a conversion around the `?:`); the inner `?:` has 3 | ✅ | p14, d01 #5 |
| 3.6 | `std::meta::operators` gains the conditional operator | ❌ no enumerator; `is_operator_expression` is false on `?:` | ❌ same | p14b |
| 3.6 | `is_prefix` / `is_postfix` | ✅ | ✅ | p15 |
| 2.5 | the `update` example: `p.x += p.y * 0.5` as an expression statement with `+=` | ✅ | ❌ `+=` is lost (see `expression_operator_of`) | p13, d01 #7 |
| 5.1 | `build()` as printed (`namespace m = std::meta`) | ✅ | ✅ | p17 |

## Open gaps

All of these are CompilerGap issues.

- #237: a null statement is an expression statement on neither fork (p21). The
  fix is gcc#24, gcc-darwin-arm64#21 and clang-p4395#9, in integration PR #244.
- #245: a nested block that declares nothing isn't a statement on GCC (p04).
- #246: GCC lowers a compound assignment (p13). #213 and #218 cover the AD
  engine's support for assignments, not this reflection shape.
- #247: on GCC, a loop is a compound statement of lowered pieces (p24).
- #248: on GCC, a class-type argument passed by value has no operands (p23). The
  engine can't differentiate through such an operator on GCC.
- #249: an operand naming a reference is a conversion on GCC and an
  id-expression on clang (p22).
- #250: on GCC, what's left of an `if constexpr` depends on the rest of the body
  (p03). A body that is a taken `if constexpr` differentiates on GCC and gets
  "unsupported statement" on clang.
- #232: equality and mangling of expression reflections differ between the
  forks. P4395R0 doesn't say which is right.
- #242: on GCC, `constant_of` of an id-expression throws, where clang gives the
  named variable's constant. It's outside the paper's API. The fix is gcc#23 and
  gcc-darwin-arm64#20, also in #244.

`tests/known_gap/gcc/ubsan_reflection_repro.cpp` is also still a gap at this
pin.

## Fixed since this report's first version (2026-10-25)

This lists the fixes that changed this report's rows or the shape of a body, not
every closed CompilerGap issue.

| Gap | Fixed by | Reached `main` in |
|---|---|---|
| Six GCC reflection bugs, from `fabs` to an assignment reported as a declaration (#183–#188) | gcc#6–#8 | #192 |
| GCC lacked 12 of the 24 functions, and its `callee_of`, call `operands_of`, leaves (#202), `body_of` shape, and statements that were also expressions all differed; `is_statement` could crash the compiler (#201) | gcc#11 | #227 |
| A local aggregate or array with a non-constant initializer (#194) | gcc#12 | #211 |
| A constructed local is one declaration (#215) | gcc#16 | #229 |
| Several declarators are one statement (#220) | gcc#17 | #231 |
| A statement reflection as a template argument was an ICE (#225) | gcc#18 | #233 |
| A static local's dynamic initialization (#230) | gcc#19 | #234 |
| A nested block that declares something, and a `using` alias, are statements (#235) | gcc#20 | #241 |
| Module importers lost a body's statement details (#239) | gcc#21 | #241 |
| Operators are reported as written, not folded, though `+=` is still lowered (#224; #246) | gcc#22 | #241 |
| `has_body` on clang (#204); a structured binding isn't a simple declaration on clang (#203) | clang-p4395#8 | #241 |

## Fork API beyond the paper

R1 would need to adopt or rename these. Both forks already ship them.

| | clang | GCC |
|---|---|---|
| `?:` | `is_conditional_operator`, `condition_of`, `true_expression_of`, `false_expression_of` | same |
| conversions / construction | `is_conversion`, `is_construct` | same, plus `is_cast` |
| per-variable access | `declared_variable_of`, `initializer_of` | same, plus `is_variable_reference` (an alias of `is_id_expression`) |
| statements §4.1 left out | `is_if_statement`, `is_for_statement`, `is_while_statement`, `is_range_for_statement`, `init_statement_of`, `then_statement_of`, `else_statement_of`, `increment_of`, `loop_body_of`, `range_init_of`, `loop_variable_of` | — |
| operator predicates | — | `is_unary_operator`, `is_binary_operator` |

## Paper-text items for R1

- §3.5's `max` example `static_assert`s fail on the clang fork, because of the
  implicit conversion. The paper either needs to say whether implicit
  conversions are visible as operands, or needs a different example. The forks
  disagree on where these nodes appear, so R1 needs a rule rather than a remark.
- §3.6 proposes an enumerator for `?:`; both forks use dedicated accessors
  instead.
- §3.5 leaves member access to "a future revision", but §2.5's `update` example
  relies on it. Both forks return an expression no predicate recognises, and
  they disagree on its operands: `p.x` is `{p}` on clang and `{p, x}` on GCC
  (d01 #7).
- §3.5 doesn't say what `expression_of` gives for a null statement, an
  expression statement with no expression (#237).
- The paper doesn't define equality of expression reflections (#232).
- §5 links `siqiyyyy/clang-p2996` (`expr-reflect`) and
  `yesmanchyk/gcc-mirror` (`autograd-capable`). The maintained forks are
  `CppReflectionAD/clang-p4395` and `CppReflectionAD/gcc` (`p4395`).

## Reproducing

The probes are not part of the test suite; nothing builds `docs/`. They need
the fetched compilers in `build/` (`python3 run_tests.py --fetch-compilers`,
see [CONTRIBUTING.md](../CONTRIBUTING.md)). In a worktree without its own
`build/`, set `CLANGXX` and `GXX` to the main checkout's compilers.

```sh
docs/p4395r0-fork-gaps/run_probes.sh                   # every p*.cpp: PASS/FAIL per compiler
docs/p4395r0-fork-gaps/run_probes.sh p14               # only probes starting with p14
docs/p4395r0-fork-gaps/compile.sh d01_dumps.cpp -DWHICH=5   # the tree each fork returns for case 5
docs/p4395r0-fork-gaps/compile.sh p24_loop_statement.cpp    # one probe, with its diagnostics
```

Each probe asserts the paper's behaviour, so a FAIL marks a gap, except p02 and
p20, which only record what the forks do where the paper promises nothing. A
probe that passes on both forks stays as a guard against regressions.
`d01_dumps.cpp` prints through a failing `static_assert`. The scripts cap memory
at about 6 GB.
