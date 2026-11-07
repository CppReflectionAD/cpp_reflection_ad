# What a reflected loop looks like on GCC

Could GCC's `condition_of` accept an `if` or `while` statement, as clang's
does? This records what the GCC fork hands back for each. Measured on
2026-11-07, using CI's build of `CppReflectionAD/gcc` at `458ffb3`, `main`'s
pin, in the `gcc-mirror` container (see
[pinned-gcc-container.md](pinned-gcc-container.md)).

## Summary

- **An `if` survives, and `condition_of` already reads it.** GCC lowers an
  `if` to the same `COND_EXPR` node as `?:`, so `condition_of`,
  `true_expression_of` and `false_expression_of` work on a reflected `if`
  with no fork change. An `if` with no `else` gives a `void` conversion from
  `false_expression_of`, not a null reflection. `statements_of` also gives
  the condition, then-branch and else-branch. GCC has no `is_if_statement`;
  `ad::meta` supplies one (`include/ad/meta.h`), true for both a written `if`
  and a loop's lowered `if … goto`.
- **A `while` or `for` doesn't.** By the time reflection sees the body, the
  loop has been lowered to labels and `goto`s, and the condition survives only
  inside an `if (cond) goto body; else goto exit;`. There is no `WHILE_STMT`
  or `FOR_STMT` left for `condition_of` to read, though `condition_of` on
  that lowered `if … goto` does give the loop condition. This is #247 (in
  [p4395r0-fork-gaps.md](p4395r0-fork-gaps.md), "On GCC, a `for` or `while`
  satisfies `is_compound_statement`").
- **Neither templates nor `constexpr` help.** A `while` in a function template
  instantiation or in a `constexpr` function is lowered the same way.
- **A local with a destructor adds a hidden `try`/`finally`.** GCC wraps the
  rest of the block in it and calls the destructor in the `finally`;
  `statements_of` walks into the `try` part and skips the `finally`, so the
  block reads as written.

clang's `condition_of` already accepts `if`, `for` and `while`
(`clang-p2996/clang/lib/AST/ExprConstantMeta.cpp`), and clang has
`is_if_statement`, `is_while_statement` and the other control-flow accessors.
GCC has none of them.

## Why the loop is gone

The C++ front end lowers loops while genericizing a function, before
`c_genericize` (`gcc/gcc/c-family/c-gimplify.cc`) writes the
`-fdump-tree-original` dump; its comment says "The C++ front end has already
done this in lang-specific code". `body_of` reflects that same lowered
`DECL_SAVED_TREE`, so the dump shows exactly what reflection sees. Supporting
`condition_of` on a loop would mean the fork keeping a copy of the body from
before the lowering and pointing `body_of` at it.

`operands_of` doesn't help either: on GCC it only accepts expressions.
`statements_of` is what splits an `if` or a lowered loop into its parts.

## Reconstructing a loop from the lowered form

The lowered pieces can be recognized through the API:

| Construct | How it reflects |
|---|---|
| `goto L` | `is_expression`, one operand; `declaration_of` gives the label, which `display_string_of` prints as `<label_decl>` |
| `L:` | `is_statement`, no children; `declaration_of` gives the label |
| `if (c) goto …; else goto …;` | `condition_of` gives `c`; `true_expression_of` and `false_expression_of` give the `goto`s |

Each loop has a fixed shape:

```
while (c) B;      [goto C, L_body:, B…, L_cond:, if (c) goto L_body; else goto L_exit;, L_exit:]
do B; while (c);  [L_body:, B…, if (c) goto L_body; else goto L_exit;, L_exit:]
for (I; c; N) B;  { I, then the while shape with N just before L_cond: }
break;            if (cond) goto L_exit;   (an `if` with no else)
continue;         goto L_cond;
```

So `while`, `do`/`while` and `for` without `break` or `continue` can be
recognized by shape alone. Some sources lower to the same tree: `for (I; c;
N) B` and `{ I; while (c) { B; N; } }` match, which is harmless because they
behave the same. Hand-written `goto`s and labels would match too.

`break` and `continue` can't be told apart without knowing which label a
`goto` targets, and labels can't be compared:

- A label has no identifier, and every label in a loop reports the loop's
  line as its `source_location_of`.
- `==` on two reflections of the same label is true, but on two different
  labels it is an internal compiler error in `cp_tree_equal`
  (`gcc/gcc/cp/tree.cc:4577`): a `LABEL_DECL` is a `tcc_declaration`, which
  that function doesn't handle, so it reaches `gcc_unreachable ()`. The same
  ICE also came up when comparing every pair of nodes in a lowered body; which
  pair set it off wasn't narrowed down, so it may not be limited to labels.

Commits on the `conditional-statements` branch of `CppReflectionAD/gcc`
(not yet in the pin) add `std::meta::is_if_statement`, `is_goto_statement`,
`is_label_statement` and `label_of`, and make two different labels compare
unequal instead of ICEing, which is option 2 below. With them, `goto1.C` in
the fork's testsuite matches a lowered loop's gotos to its labels and tells a
`break` from the loop's condition.

The options, cheapest first:

1. Match the shapes above in `include/ad/meta.h`, for loops with no `break`
   or `continue`. A change to how GCC lowers loops would break it, so tests
   need to pin the shapes.
2. Fix the ICE in the fork so that different labels compare unequal. That
   makes `break` and `continue` resolvable, and is a far smaller change than
   #247.
3. Fix #247: give reflection the body from before loops are lowered, so GCC
   has real loop statements, as clang does.

## The measurements

A throwaway test in `tests/gcc_only/` walked each body with `statements_of`
and printed the predicates that hold on each node. It was built with
`run_tests.py --compiler gcc --tests <name> --extra-cxxflag=-fdump-tree-original`;
the dump lands next to the object file as `<name>.cpp.cpp.006t.original`.
Below, the dump is on the left and the reflection on the right.

### `while`

```cpp
double w(double x) {
  double y = x;
  while (y < 10.0)
    y = y * 2.0;
  return y;
}
```

```
double y = x;                                     stmt decl
                                                  stmt compound
goto <D.70674>;                                     expr
<D.70675>:;                                         stmt (label)
(void) (y = y * 2.0e+0);                            stmt exprstmt
<D.70674>:;                                         stmt (label)
if (y < 1.0e+1) goto <D.70675>;                     stmt -> [expr op, expr, expr]
  else goto <D.70673>;
<D.70673>:;                                         stmt (label)
return <retval> = y;                              stmt return
```

A braced body with two statements, the same loop in a function template
instantiation, and the same loop in a `constexpr` function all give this
shape.

### `for`

```cpp
for (int i = 0; i < 3; ++i)
  y = y * 2.0;
```

```
{
  int i = 0;
  goto <D.70684>;
  <D.70683>:;
  (void) (y = y * 2.0e+0);
  (void) ++i;
  <D.70684>:;
  if (i < 3) goto <D.70683>; else goto <D.70681>;
  <D.70681>:;
}
```

The same as `while`, with the init-statement in an enclosing block and the
increment before the condition label.

### `do`/`while` and `break`

```cpp
do
  y = y * 2.0;
while (y < 10.0);

while (y < 10.0) {
  if (y > 5.0)
    break;
  y = y * 2.0;
}
```

```
<D.70685>:;
(void) (y = y * 2.0e+0);
if (y < 1.0e+1) goto <D.70685>; else goto <D.70683>;
<D.70683>:;

goto <D.70691>;
<D.70692>:;
if (y > 5.0e+0) goto <D.70690>;
(void) (y = y * 2.0e+0);
<D.70691>:;
if (y < 1.0e+1) goto <D.70692>; else goto <D.70690>;
<D.70690>:;
```

A `do`/`while` has no leading `goto`. The `break` is a `goto` to the same
exit label as the loop condition's `else`.

### `if`

```cpp
double i(double x) {
  if (x > 0.0)
    return x;
  return -x;
}
```

```
if (x > 0.0)                                      stmt -> [expr op, stmt return, ...]
  {
    return <retval> = x;
  }
return <retval> = -NON_LVALUE_EXPR <x>;           stmt return
```

### A local with a destructor in the loop body

```cpp
struct G {
  double v;
  ~G() { std::puts("~G"); }
};

while (y < 10.0) {
  G g{y};
  y = g.v * 2.0;
}
```

```
goto <D.70705>;                                   expr
<D.70706>:;                                       stmt (label)
{                                                 stmt compound
  struct G g;                                       stmt decl
  g.v = y;
  try {
    (void) (y = g.v * 2.0e+0);                      stmt exprstmt
  } finally {
    G::~G (&g);                                     (not returned)
  }
}
<D.70705>:;                                       stmt (label)
if (y < 1.0e+1) goto <D.70706>;                   stmt -> [expr op, expr, expr]
  else goto <D.70704>;
<D.70704>:;                                       stmt (label)
```

The loop is lowered as before. `G g{y}` appears in the dump as a declaration
followed by `g.v = y`, but reflection returns only the declaration and then
the assignment to `y`, so it seems to count the initialization as part of the
declaration. That wasn't checked with `initializer_of`.
