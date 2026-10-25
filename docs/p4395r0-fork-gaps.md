# P4395R0 vs. the compiler forks: implementation gaps

This compares what [P4395R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4395r0.pdf)
(2026-10-23, "Reflecting Function Bodies") proposes with what the two
compiler forks do today. It covers the §3 API and its semantic promises, and
the code shown in §2 and §5. Measured on 2026-10-25.

| Column | What it is |
|---|---|
| **clang** | `CppReflectionAD/clang-p4395` at `7d2b07c`, our pin. The `p4395` tip is 16 commits ahead, but its `<meta>` statement/expression API is identical. |
| **GCC** | `CppReflectionAD/gcc` at `311c34e`: main's pin, using CI's fetched build. |
| **GCC + fixes** | A local build of gcc#6–#8, which is what #192 pins (`cd19f55`). It is shown only where it differs. The `p4395` tip differs from both pins only by renaming `return_value_of` to `return_operand_of` (gcc#4). |

Each probe in [`p4395r0-fork-gaps/`](p4395r0-fork-gaps/) is a small file
asserting one claim from the paper. It is compiled with `std::meta` exactly as
the paper spells it, and on GCC a second time through `include/ad/meta.h`.
The `ad::meta` results were measured with #198's version (`acf056c`); with
`main`'s, the `&&`/`||`/`!` row (p16) fails on GCC until #198 merges.

✅ matches the paper · ⚠️ present but behaves differently · ❌ missing or wrong

## Headlines

1. **Neither fork has `has_body`** (§3.1), not even at the fork tips. On clang,
   `body_of` on a function with no definition is a hard error. The clang fork
   has no `std::meta::exception`, so it can't be caught, and there is no way
   to ask first. On GCC, catching the exception works as a substitute.
2. **The paper's own §3.5 `?:` example fails on clang.** `operands_of(e).size()`
   is 1, not 3. Clang wraps the conditional in an implicit lvalue-to-rvalue
   conversion. It does the same to every id-expression used as a value, so
   operands such as `x` in `x * 2` fail `is_id_expression` and `declaration_of`.
   The paper has no notion of conversion nodes; the clang fork adds
   `is_conversion` for them.
3. **Neither fork extends `std::meta::operators` with `?:`** (§3.6). Both
   instead add `is_conditional_operator`, `condition_of`, `true_expression_of`
   and `false_expression_of`, and `is_operator_expression` is false on a `?:`.
4. **GCC lacks 12 of the 24 proposed functions** (11 at its fork tip); clang
   lacks only `has_body`. `ad::meta` papers over most of the renames, but GCC's
   tree shape differs from the paper's model in ways a shim can't fix:
   - statements are also expressions;
   - expression statements are not statements;
   - nested blocks are flattened;
   - multi-declarators are split into one statement per variable;
   - a variable is its own operand, so the recursive walk §3.5 describes never
     terminates;
   - `+=` is lowered to `= … + …`, and prefix vs. postfix can't be told apart.

   An overloaded operator call doesn't report its operator either, but
   `operator_of(declaration_of(callee_of(e)))` recovers it (p18), so a shim
   could.
5. **New GCC bug:** an internal compiler error (`r07`). See
   [Other fork defects](#other-fork-defects).
6. **Clang reports a structured binding as `is_simple_declaration`**, although
   the paper and the fork's own `<meta>` comment both say bindings are excluded.

## API and semantics, claim by claim

| § | Paper API / claim | clang | GCC (and + fixes) | Probe |
|---|---|---|---|---|
| 3.1 | `has_body` | ❌ absent | ❌ absent | p01 |
| 3.1 | `body_of` on a declared-only function | ⚠️ not a constant expression; can't be caught, because the fork has no `std::meta::exception` | ⚠️ throws `std::meta::exception` "function is not defined"; catchable, so it can stand in for `has_body` | p02, p20 |
| 3.1 | `body_of` gives a compound statement | ✅ | ⚠️ a single-statement body is that statement (e.g. the `return`). A body with locals is a scope node that satisfies `is_expression` and has 3 `operands_of` | d01, r06, r07 |
| 3.2 | `statements_of` lists the block's statements in order | ✅ | ⚠️ nested `{ … }` blocks are flattened into the parent list; a `for` comes back as a lowered statement list; a `using` alias disappears | p04, d01 #2 #4 |
| 3.2 | `is_compound_statement` | ✅ | ❌ absent | p04 |
| 3.2 | every `statements_of` element satisfies `is_statement` | ✅ | ❌ an expression statement comes back as a void cast expression with `is_statement` false | p03, r02 |
| 3.2 / 4.2 | the discarded `if constexpr` branch is not returned | ✅ its statements are gone. ⚠️ The `if` stays, and its `else_statement_of` is an empty compound, not null | ✅ the `if` is folded away, leaving only the taken branch | p03, d01 #1 |
| 3.2 | `is_declaration_statement` | ✅ | ⚠️ at `311c34e`, `p.x += …` is reported as a declaration (#185, fixed by gcc#6). A structured binding is not a declaration statement on either GCC | p06, d01 #4 #7 |
| 3.2 | `is_expression_statement` | ✅ | ❌ absent | p04, p13 |
| 3.2 | `is_return_statement` | ✅ | ✅ (also `is_expression`) | p04 |
| 3.3 | `is_simple_declaration` | ⚠️ **true** for `auto [a, b] = p;`, which should be excluded; correctly false for `using D = double;` | ❌ absent | p06 |
| 3.3 / 5.1 | `declarators_of` / `initializers_of` (`double v = …, w = …;` gives 2 of each) | ✅ | ❌ absent: GCC has `declared_variable_of` + `initializer_of`, and splits a multi-declarator into one statement per variable. `ad::meta` returns one-element vectors | p05, r02 |
| 3.4 | `return_operand_of`, including `return {x, y};` → the constructor call | ✅ | ⚠️ called `return_value_of` at both pins (renamed only at the fork tip). Through `ad::meta` it behaves as the paper says, braced returns included | p12 |
| 3.5 | `expression_of` | ✅ | ❌ absent; an expression statement already *is* its expression | p13 |
| 3.5 | `is_expression` | ✅ statements aren't expressions | ⚠️ also true for declaration statements, return statements and function bodies | r01, r06 |
| 3.5 | `is_operator_expression` (built-in or overloaded, including `&&`, `\|\|`, `!`) | ✅ | ❌ absent: GCC has `is_unary_operator` / `is_binary_operator`, which leave out `&&`, `\|\|`, `!` and `=`. `ad::meta` adds the logical operators (#198) | p10, p11, p16 |
| 3.5 | `expression_operator_of` | ✅ | ⚠️ an overloaded `a + b` reports `()`, not `op_plus`, though `operator_of(declaration_of(callee_of(e)))` gives `op_plus`. `p.x += e` is lowered to `p.x = p.x + e`, so `op_plus_equals` never appears | p10, p18, p13, d01 #6 #7 |
| 3.5 | `is_function_call`, also true for an overloaded operator | ✅ | ✅ | p08, p10 |
| 3.5 | `callee_of` is the raw callee expression; `is_id_expression` + `declaration_of` resolve it | ✅ | ❌ returns the function itself (`operands_of(e)[0]`), not an expression; `ad::meta::declaration_of` passes it through | p08 |
| 3.5 | `operands_of` on a call: the arguments, defaulted ones included | ✅ | ⚠️ the callee is listed first (`ad::meta` drops it); defaulted arguments are included | p09 |
| 3.5 | operands are leaves or sub-expressions to recurse into | ✅ | ❌ `operands_of(v)` for a variable is `{v}`, so a walk that recurses into every `is_expression` operand never ends | r05, p17b |
| 3.5 | `is_literal` | ✅ | ✅ | p11 |
| 3.5 | `is_id_expression` | ⚠️ exists, but value uses arrive wrapped in an implicit conversion, so `operands_of(v + 1)[0]` is not an id-expression | ❌ absent: `is_variable_reference` is just `is_variable`, because the operand *is* the variable reflection. `ad::meta` maps it | p07, p09 |
| 3.5 | `declaration_of(id)` matches `declarators_of(decl)[i]` | ⚠️ only after stripping the implicit conversion (`while (is_conversion(e)) e = operands_of(e)[0];`); on the operand as returned it is not a constant expression | ✅ through `ad::meta` | p07, p19 |
| 3.5 | the `max` example: `operands_of(e).size() == 3` | ❌ size is 1 (a conversion around the `?:`); the inner `?:` has 3 | ⚠️ doesn't compile as written (no `return_operand_of`); the `?:` node itself has 3 operands | p14, d01 #5 |
| 3.6 | `std::meta::operators` gains the conditional operator | ❌ no enumerator; `is_operator_expression` is false on `?:` | ❌ same | p14b |
| 3.6 | `is_prefix` / `is_postfix` | ✅ | ❌ absent, and `++i` and `i++` both report `op_plus_plus` with nothing else to tell them apart | p15, d01 #8 |
| 2.5 | the `update` example: `p.x += p.y * 0.5` as an expression statement with `+=` | ✅ | ❌ `+=` is lost (see `expression_operator_of`) | p13, d01 #7 |
| 5.1 | `build()` as printed (`namespace m = std::meta`) | ✅ | ❌ doesn't compile (`declarators_of`, `initializers_of`, `return_operand_of`). Spelled `ad::meta`, it compiles, but the recursion exceeds the constexpr depth limit of 512 (see the leaves row) | p17, p17b |

## Other fork defects

- **GCC internal compiler error (new, not filed).** In a function with a
  local, `operands_of(body_of(f))` has 3 elements. The third is a raw scope
  block, and `is_statement` on it crashes the compiler: `tree check: expected
  tree that contains 'typed' structure, have 'block' in error_operand_p`. It
  happens on both `311c34e` and the gcc#6–#8 build. The paper never calls
  `operands_of` on a body, but GCC accepts the call because the body satisfies
  `is_expression`. Repro: `r07_gcc_body_operands_ice.cpp`.
- **Open CompilerGap issues, all GCC.** #183 (`fabs`), #184 (`const T&`
  return), #185 (assignment as a declaration), #186 (class local is two
  declarations), #187 (lost initializer) and #188 (NRVO return) are fixed by
  gcc#6–#8. #192 picks those up. At `7e218fb` (checked 2026-10-25), every
  test, sanitize and archive job on #192 passed. Only `Submodule pins` is red,
  until the fork PRs merge. #194 (non-constant aggregate or array
  initializer) is still open and unfixed.
- **GCC + UBSan:** `tests/known_gap/gcc/ubsan_reflection_repro.cpp`.

## Fork API beyond the paper

R1 would need to adopt or rename these. Both forks already ship them.

| | clang | GCC |
|---|---|---|
| `?:` | `is_conditional_operator`, `condition_of`, `true_expression_of`, `false_expression_of` | same |
| conversions / construction | `is_conversion`, `is_construct` | `is_cast`, `is_construct` |
| per-variable access | `declared_variable_of`, `initializer_of` | same, plus `is_variable_reference` |
| statements §4.1 left out | `is_if_statement`, `is_for_statement`, `is_while_statement`, `is_range_for_statement`, `init_statement_of`, `then_statement_of`, `else_statement_of`, `increment_of`, `loop_body_of`, `range_init_of`, `loop_variable_of` | — |
| operator predicates | — | `is_unary_operator`, `is_binary_operator` |

## Paper-text items for R1

- §3.5's `max` example `static_assert`s fail on the clang fork, because of the
  implicit conversion. The paper either needs to say whether implicit
  conversions are visible as operands, or needs a different example.
- §3.6 proposes an enumerator for `?:`; both forks use dedicated accessors
  instead.
- §5 links `siqiyyyy/clang-p2996` (`expr-reflect`) and
  `yesmanchyk/gcc-mirror` (`autograd-capable`). The maintained forks are
  `CppReflectionAD/clang-p4395` and `CppReflectionAD/gcc` (`p4395`).

## Reproducing

The probes are not part of the test suite; nothing builds `docs/`. They need
the fetched compilers in `build/` (`python3 run_tests.py --fetch-compilers`,
see [CONTRIBUTING.md](../CONTRIBUTING.md)). In a worktree without its own
`build/`, set `CLANGXX` and `GXX` to the main checkout's compilers. To add a
second GCC column (such as a build of fork PRs), set `GXX_FIXED`.

```sh
docs/p4395r0-fork-gaps/run_probes.sh                  # every p*.cpp: PASS/FAIL per configuration
docs/p4395r0-fork-gaps/run_probes.sh p14              # only probes starting with p14
docs/p4395r0-fork-gaps/compile.sh d01_dumps.cpp -DWHICH=5               # the tree each fork returns for case 5
docs/p4395r0-fork-gaps/compile.sh r07_gcc_body_operands_ice.cpp -DWHICH=4   # the GCC ICE
docs/p4395r0-fork-gaps/compile.sh r02_gcc_block.cpp -DFN=nested -DWHICH=1   # GCC statement shapes
```

Each probe asserts the paper's behaviour, so a FAIL marks a gap, and every
probe fails on at least one compiler. `d01_dumps.cpp` prints through a failing
`static_assert`. The scripts cap memory at about 6 GB. Without the cap, a
recursive walk over GCC's self-referencing operands used 26 GB before it was
killed.
