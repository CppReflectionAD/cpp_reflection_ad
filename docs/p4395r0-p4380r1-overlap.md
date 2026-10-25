# P4395R0 and P4380R1: overlaps and how to address them

This note compares two WG21 papers from the October 2026 mailing:

- [P4395R0] *Reflecting Function Bodies: Expression and Statement
  Introspection* (SG7, 2026-10-23). This is our paper, and the library in this
  repository is its demonstration. It proposes **read-only** reflection of
  function bodies: `body_of`, `statements_of`, declaration and return
  statements, and expressions down to operators and operands. It defers
  expression splicing to a later paper (§4.3).
- [P4380R1] *Token Sequence Injection* (EWG, 2026-10-22), by Barry Revzin,
  Daveed Vandevoorde and Andrei Alexandrescu. It follows up [P3294R2] and
  proposes **code generation**: `^^{ ... }` token sequences,
  `queue_injection`, annotation hooks and declaration cloning. It also proposes
  token sequence *macros* (`__macro`) whose typed parameters are reflections
  of the argument expressions.

The papers are mostly complementary. P4395R0 reads code and P4380R1 writes it.
But P4380R1's macros need to look at expressions, so they bring their own
expression-reflection API (§4.1, "Expression macros"). That API overlaps
P4395R0 §3.5 and sometimes conflicts with it. Neither paper cites the other.

Section numbers below are each paper's own. "Our forks" means the clang and
GCC forks that this repository pins as submodules.

## Summary

| # | Overlap | Kind | Suggested resolution |
| - | ------- | ---- | -------------------- |
| 1 | Both papers reflect an expression as a `std::meta::info` | shared foundation | One expression-reflection kind, specified once |
| 2 | `declaration_of(info)` | name clash, different meaning | Rename one of the two (we suggest ours) |
| 3 | `operator_of(info expr)` vs `expression_operator_of(info)` | same query, different name | Agree on one name, with the reasons on record |
| 4 | `operands_of(info)` | same name, unspecified or different semantics | Specify calls, default arguments and the callee once |
| 5 | `is_binary_operation` vs `is_operator_expression`, `is_constant_expression` vs `is_literal` | overlapping predicates | Keep both sets where they mean different things; define them together |
| 6 | `type_of`, `source_text_of`, `source_location_of` on expressions | in P4380R1 only, useful for P4395R0 §2.2 | Apply them to every expression reflection |
| 7 | `std::meta::operators` | P4395R0 extends it, P4380R1 interpolates it as tokens | Decide how `?:` and cast enumerators interpolate |
| 8 | Expression splicing | P4395R0 defers it, P4380R1 interpolation provides a form of it | Make P4380R1 the write side; solve relocation |
| 9 | Assertions, short-circuiting, test generation | overlapping motivating examples | Present them as one story: macro at the call site, body reflection below it |
| 10 | What an expression reflection shows (conversions, instantiation, discarded statements) | interacting design choices | Agree on the abstraction level |

## 1. One kind of expression reflection

In P4380R1 §3, a macro's typed parameter, such as `cond` in
`template <class T> __macro check(T&& cond)`, is "a reflection representing an
expression". In P4395R0 §3.5, `expression_of`, `return_operand_of`,
`initializers_of` and `operands_of` return reflections that satisfy
`is_expression`.

The two papers describe the same thing: a semantically analysed expression,
reflected as an `info`. They should share a definition. A query from either
paper would then work on an expression from the other: a macro argument
could be walked with P4395R0's API, and a body expression could be
interpolated with P4380R1's. Sections 2 to 7 list the places where the two
surfaces would have to agree first.

The table puts the expression APIs side by side.

| Purpose | P4395R0 (§3.5, §3.6) | P4380R1 (§4.1) |
| ------- | -------------------- | -------------- |
| Is this an expression? | `is_expression(info)` | implied for macro parameters |
| Top-level operator | `is_operator_expression(info)`, `expression_operator_of(info) -> operators` | `is_binary_operation(info)`, `operator_of(info) -> operators` |
| Call | `is_function_call(info)`, `callee_of(info)` | — |
| Operands | `operands_of(info) -> vector<info>` | `operands_of(info) -> vector<info>` "as written, each interpolable" |
| Leaves | `is_literal`, `is_id_expression`, `declaration_of(info)` | `is_constant_expression(info)` |
| Prefix or postfix | `is_prefix`, `is_postfix` | — |
| Type | — | `type_of(info e)`, "`decltype(arg as written)`" |
| Source | — | `source_text_of(info e)`, `source_location_of(info e)` |
| Value category | — | `as_lvalue(info e)` |
| Context | — | `macro_expansion_context()` |

## 2. `declaration_of`: same name, different operation

This is the most serious clash.

- **P4395R0 §3.5:** `consteval info declaration_of(info);` takes an
  id-expression and returns the entity it names. For example,
  `declaration_of(callee_of(e))` gives the called function.
- **P4380R1 §2.7, §4.1:**
  `consteval auto declaration_of(info fn, clone_naming = {}) -> info;` takes a
  function or function template and returns a *declaration spec*
  (`is_declaration_spec`). A declaration spec is a cloned declaration that can
  be interpolated into a token sequence, renamed, and transformed with
  `make_override` and `make_noexcept`.

Both take an `info` and return an `info`, and P4380R1's second parameter has a
default, so a call `declaration_of(r)` could mean either one. Overloading on the
kind of `r` (an expression or a function) is fragile:

- The two meanings chain naturally, so the same spelling would appear
  twice with two meanings. In `declaration_of(declaration_of(callee_of(e)))`,
  the inner call resolves the callee and the outer call clones it.
- Our own compatibility layer already blurs the input kinds.
  [`ad::meta::declaration_of`](../include/ad/meta.h) passes a function
  reflection through unchanged, because the GCC fork's `callee_of` returns the
  function itself rather than an expression that names it. Under P4380R1, a
  function reflection is exactly the input that `declaration_of` clones.

The library calls the P4395R0 meaning in four places
([`autograd.h`](../include/ad/autograd.h) lines 395, 478, 549 and 610).

**How to address it.** One of the papers should rename. We suggest P4395R1
does, because P4380R1 is further along (EWG) and its name describes its
result, a declaration. The names below are first ideas; none has been checked
for clashes:

- `referenced_entity_of(info)` or `entity_of(info)` says what P4395R0 returns:
  the entity, not a declaration of it.
- If P4380R1 renames instead, `clone_declaration(info, clone_naming)` matches
  what it does.

Whichever paper renames, the other should mention the rename so that the
committee sees one consistent vocabulary.

## 3. `operator_of` vs `expression_operator_of`

The papers make opposite choices for the same query.

- **P4395R0 §3.5** adds `expression_operator_of` "rather than overload
  `operator_of` from P2996, since they have distinct semantics despite both
  relating to operators". P2996's `operator_of` takes an operator *function*.
- **P4380R1 §3.2, §4.1** overloads `operator_of(info e)` for expressions and
  also adds `operator_of(token)`.

P4380R1's choice reads well in `check!`: `\(operator_of(cond))`. And for an
overloaded `a + b`, the expression's operator and `operator_of` of the callee
are the same value. P4395R0's reason still applies to built-in operators,
which have no operator function, and to the operators P4395R0 §3.6 adds that
P2996 has no function for (`?:`, casts).

**How to address it.** Make one decision, in one paper, and have the other
follow it. If `operator_of` is overloaded, specify it once for every kind of
input: operator function, expression and token.

## 4. `operands_of`: what is an operand?

Both papers use the name, and they describe it differently.

- **P4395R0 §3.5:** for an operator or a call, the arguments. A call with
  default arguments gets "one operand for each function parameter, including
  defaulted arguments". The callee is separate (`callee_of`).
- **P4380R1 §4.1:** "as written, each interpolable". The paper only uses it on
  binary operations (`check!`).

"As written" and "including defaulted arguments" disagree on `f(a)` when `f`
has a second, defaulted parameter. "Each interpolable" also matters: a
defaulted argument has no tokens at the call site, so interpolating it into
another context is a question P4380R1 hasn't had to answer.

Our forks show that the callee question isn't hypothetical. The GCC fork lists
the callee as the first operand of a call, and
[`ad::meta::operands_of`](../include/ad/meta.h) removes it so both compilers
agree.

**How to address it.** Specify `operands_of` for every expression kind in one
place. That means calls with and without default arguments, member calls (is
the object expression an operand?), `?:`, and overloaded operators. The
specification should also say what interpolating a defaulted operand means. If
"as written" is the macro need and "per parameter" is the analysis need, two
functions may be cleaner than one with a mode.

## 5. Overlapping predicates

| P4395R0 | P4380R1 | Relationship |
| ------- | ------- | ------------ |
| `is_operator_expression` (unary, binary, `?:`, overloaded operators) | `is_binary_operation` | P4380R1's is a narrower case. `is_operator_expression(e) && operands_of(e).size() == 2` gives it, apart from overloaded operators (see below). |
| `is_literal` (a literal token) | `is_constant_expression` ("`constant_of(e)` would succeed") | Different questions. `n + 1` with `constexpr int n` is a constant expression but not a literal. Both are useful. |
| `is_function_call` | — | P4380R1's `check!` falls back to whole-expression output for calls. |

Overloaded operators need one rule in both papers. P4395R0 §3.5 says
`a + b` with a user-defined `operator+` satisfies both `is_function_call` and
`is_operator_expression`. On our GCC fork it is only a call (see the
`ad::meta` notes in [`meta.h`](../include/ad/meta.h)). For `check!`, the
rule decides whether `x == y` on a user type gets decomposed.

## 6. Queries P4380R1 has that P4395R0 lacks

P4380R1 §4.1 has two queries on expressions that P4395R0 §3 doesn't:

- `source_text_of(info e) -> string_view`
- `source_location_of(info e) -> source_location`

P4395R0 §2.2's `MY_ASSERT` gets by without them. Its [Godbolt demo] takes
the text from `#__VA_ARGS__` and the line from a defaulted
`std::source_location` argument, then reprints each subexpression from the
reflected tree. With these queries, it could show each subexpression as the
user spelled it and where it is, instead of a reconstruction.

P4380R1 also has `type_of(info e)`. P4395R0 doesn't say whether P2996's
`type_of` applies to expression reflections. Body-walking libraries need it,
for example to handle subexpressions whose type isn't `double`.

**How to address it.** P4395R1 should say that these apply to every
expression reflection, not only macro parameters, and cite P4380R1 for them.
`source_text_of` on an expression inside a template instantiation, or one that
`if constexpr` produced, needs a definition. P4395R0 §4.2 reflects bodies
after instantiation.

## 7. `std::meta::operators`: extended by one paper, interpolated by the other

- P4395R0 §3.6 proposes extending `std::meta::operators` to "all possible
  operators, such as the conditional operator and eventually cast operators".
  It uses `is_prefix` / `is_postfix` rather than new enumerators for `++a`
  and `a++`.
- P4380R1 §2 interpolates an `operators` value as the token that spells it
  (`\(std::meta::operators::op_equals_equals)` gives `==`). §2.10 and §3.2
  rely on this to build expressions such as `+lhs $op +rhs`.

P2996's enum already has enumerators that aren't one token
(`op_parentheses`, `op_square_brackets`, `op_array_new`). So token count
isn't the new problem. The new problem is shape:

- `?:` wraps around its operands, so `lhs \(op) rhs` can't express it.
- A cast enumerator would need the target type as well. `static_cast`
  also needs the type in angle brackets around the operand.

**How to address it.** Whichever paper goes second should say how these
enumerators interpolate. They could be declared not interpolable, or
interpolation of `operators` could be defined per enumerator. P4395R0's choice
of `is_prefix` / `is_postfix` over new enumerators keeps `++` and `--` working
with interpolation as it is today.

## 8. Expression splicing: P4380R1 as P4395R0's write side

P4395R0 §4.3 deliberately stops at inspection: "Splicing relocates code from
where it was written to a different context, and that relocation raises a
distinct cluster of design questions". Two P4395R0 use cases need the write
side:

- §2.5 HPC: rewriting `update(Particle&)` into a struct-of-arrays kernel;
- §2.4 mutation testing: emitting each mutant.

P4380R1 already has expression interpolation. Interpolating an `info` that
represents an expression gives "a token whose meaning is the expression"
(§3), so `id!(1 + 2) * 3 == 9`. P3294R2 argued that code generation should
use token sequences, not AST construction. P4395R0 §4.1 makes the matching
choice for reading: stay away from the compiler's AST. So the two papers fit
together: read with semantic reflection and write with tokens. The follow-up
paper that P4395R0 §4.3 promises can be phrased as "interpolating a P4395
expression reflection into a P4380 token sequence".

### What still has to be solved: relocation

A P4380R1 macro argument is evaluated at the macro's call site, where every
name in it is in scope. P4380R1 diagnoses a second interpolation of the same
argument ("would evaluate this argument more than once", §3.3). An expression
from `body_of(f)` is different:

- It names `f`'s parameters and locals, which don't exist at the injection
  site.
- Interpolating it more than once is normal. A derivative or a mutant
  re-emits the same subexpression many times. These are new evaluations, not
  repeated evaluations of one argument.

Options, roughly from least to most ambitious:

1. **Don't splice; regenerate.** Lower the expression into your own
   representation and emit fresh tokens for it. Leaves are interpolated as
   entities (`\(callee)`, `\(type)`) and constants as values. This needs
   nothing beyond the two papers, and the AD library already works this way
   (next section).
2. **Splice with an explicit rebinding.** Interpolate a body expression with a
   map from the original function's parameters and locals to token sequences
   for the target context. For example, `x` becomes `p0` of a declaration
   cloned with P4380R1's `declaration_of(fn, {.parameter_prefix = "p"})`.
   This needs reliable identity between `declaration_of(id-expr)` (P4395R0)
   and `parameters_of(fn)`. On our clang fork they are different reflection
   kinds that never compare equal, so the library matches parameters by name
   ([`autograd.h`](../include/ad/autograd.h) around line 334).
3. **Re-lex the source.** `tokenize(source_text_of(e))` is available today
   from P4380R1 but loses hygiene and overload resolution. It is the
   [Tangent] failure mode that P4395R0 §2.1 warns about.

Option 2 is the one to design. It is the "distinct cluster of design
questions" that P4395R0 §4.3 names.

## 9. The motivating examples overlap

| Use case | P4395R0 | P4380R1 | Together |
| -------- | ------- | ------- | -------- |
| Richer assertions | §2.2 `MY_ASSERT` reports each subexpression of the asserted expression. The [Godbolt demo] gets the expression by wrapping it in a lambda inside a preprocessor macro and reflecting the lambda's body. | §3.2 `check!` decomposes one top-level comparison at the call site, as Catch2 does. | `check!` receives the expression as an `info`, which is what P4395R0 §1.1 notes P2996 deferred (`^^(expr)`), with no lambda needed. It can then walk the expression, and continue into a callee with `body_of(referenced_entity_of(callee_of(cond)))`, using P4395R0's API. Neither paper can do both levels alone. |
| Short-circuiting | §2.3 ORMs: overloading `&&` / `||` loses short-circuiting, so reflect the expression instead. | §3.7 macro operators: `__macro operator or` keeps short-circuiting. | Different problems with the same root cause. Each paper can cite the other as the solution to the other half. |
| Generated tests | §2.4 mutation testing. | §2.3 literal testing generates test calls. | Mutation testing needs both papers: read the body, then emit the mutant. |
| Library implementations of compiler features | §2.1 AD as a library. | §2.6 `Eq` as a metaclass, §2.7 `LoggingVector`. | The same pitch: features that today need a compiler plugin become `consteval` libraries. |

## 10. Abstraction level: decisions that affect both papers

- **Implicit conversions.** P4380R1 §3.2 takes `check`'s parameter as `T&&`
  rather than `bool` "because we don't want to have to walk up through that
  boolean conversion in the expression". That assumes conversions show up in
  an expression reflection. P4395R0 doesn't say. Member access and casts are
  "future revision" (§3.5). Our forks expose `is_cast` and `is_construct`,
  and the library treats both as conversions (`ad::meta::is_conversion`).
  Both papers should say the same thing about implicit conversions.
- **When the body is visible.** P4395R0 §4.2 reflects bodies after template
  instantiation, without discarded `if constexpr` branches. P4380R1 injects
  during class definition (`inject_members`, `on_complete`), and
  `macro_expansion_context()` can return a function whose body is being
  parsed. `body_of(macro_expansion_context())` inside a macro would reflect a
  body that is incomplete. Someone will try it, so one of the papers should
  say whether it's allowed.
- **Tokens vs semantics.** P4380R1's `token_sequence` macro parameters
  (`λ!`, `fun!`, `using_!`) take tokens on purpose, with no semantic
  analysis. P4395R0's case (§2.1, the Tangent comparison) is that analysis
  needs types and resolved overloads. The two levels serve different needs and
  shouldn't be merged. The typed-parameter macros are where they meet.

## 11. What this means for the AD library

`forward_derivative` doesn't generate code today. It expands a `template for`
over a static array of nodes and splices constant leaves with `[:n.leaf:]`
([`forward_derivative.h`](../include/ad/forward_derivative.h) line 25). With
P4380R1, the library could instead emit a named function: clone the
signature, then generate the body from the graph. This is option 1 in
section 8, so it needs no expression splicing:

```cpp
// Sketch only: untested, and no compiler implements both papers yet.
// declaration_of here is P4380R1's (clone), not P4395R0's (see section 2).
template <std::meta::info Fn, std::size_t Wrt>
consteval void inject_derivative(std::meta::token_sequence name) {
  auto d = std::meta::declaration_of(Fn, {.name = name}); // double name(double p0, ...)
  std::meta::token_sequence body;
  for (Node n : build(Fn)) // P4395R0 §5.1
    body += emit_tangent<Wrt>(n); // e.g. ^^{ double $t = \(cos)($v) * $dt; }
  queue_injection(^^{ \(d) { \(body) } });
}
```

The derivative would then be an ordinary named function that can be inspected
in an AST dump, as P4380R1 §2.11 does for injected code. We haven't measured
whether it compiles faster or slower than the `template for` expansion. The
two papers are implemented in different compilers (P4380R1 in the "barry
prototypes" Clang on Compiler Explorer, P4395R0 in the forks above), so
trying it first means combining the two implementations.

## Suggested actions

For P4395R1:

1. Rename `declaration_of` (section 2), or agree a rename with the P4380
   authors.
2. Cite [P3294R2] and [P4380R1] in §1.1. Say that P4380R1's macros produce
   expression reflections, and that P4380R1 is the intended write side for
   §4.3 and the §2.4 and §2.5 use cases.
3. Specify `operands_of` for calls, default arguments and the callee
   (section 4). Settle the `operator_of` naming with the P4380 authors
   (section 3).
4. Adopt `type_of`, `source_text_of` and `source_location_of` for expression
   reflections (section 6), so that §2.2 can report each subexpression as
   written.
5. Say whether expression reflections show implicit conversions
   (section 10).

For the P4380 authors (to raise with them, not to decide for them):

1. The `declaration_of` clash and the `operands_of` semantics (sections 2
   and 4).
2. How `?:` and cast enumerators of `operators` would interpolate
   (section 7).
3. Interpolating body expressions with rebinding (section 8, option 2) as a
   joint follow-up paper.

[Godbolt demo]: https://godbolt.org/z/rxMnqfav4
[P3294R2]: https://wg21.link/p3294r2
[P4380R1]: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4380r1.html
[P4395R0]: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2026/p4395r0.pdf
[Tangent]: https://github.com/google/tangent
