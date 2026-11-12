# include/nn — neural-network AD by body reflection (proof of concept)

A header-only library that differentiates tensor-level C++ (a small
multilayer perceptron, or MLP) in reverse mode. It reflects the bodies of
ordinary functions with the statement/expression reflection proposed in
*Reflecting Function Bodies: Expression and Statement Introspection*.

```cpp
#include <nn/nn.hpp>
using nn::Mat; using nn::Vec;

template <std::size_t O, std::size_t I>
Vec<O> dense_relu(const Mat<O, I> &W, const Vec<O> &b, const Vec<I> &x) {
  Vec<O> z = W * x + b;
  return relu(z);
}

double loss(const Mat<8, 2> &W1, const Vec<8> &b1, const Mat<1, 8> &W2,
            const Vec<1> &b2, const Vec<2> &x, const Vec<1> &t) {
  Vec<8> h = dense_relu(W1, b1, x);
  Vec<1> y = sigmoid(W2 * h + b2);
  return mse(y, t);
}

auto [L, g] = nn::ad::value_and_grad<^^loss, 0, 1, 2, 3>(W1, b1, W2, b2, x, t);
auto &[gW1, gb1, gW2, gb2] = g;   // each gradient has its parameter's type

// The same with an explicit rule set, where a rule listed first wins:
using E = nn::ad::engine<my_rule, nn::ad::default_rules>;
auto [L2, g2] = E::value_and_grad<^^loss, 0, 1, 2, 3>(W1, b1, W2, b2, x, t);
```

It works with both forks: Clang p2996 and GCC `-freflection`. The test is
[tests/common/nn_mlp.t.cpp](../../tests/common/nn_mlp.t.cpp). It checks
gradients against finite differences, checks the DAG's structure with
`static_assert`s, and trains a 2-8-1 MLP on XOR in a few milliseconds. Run it
with:

```sh
python3 run_tests.py --compiler both --run-executables --tests 'common/nn_mlp.t.cpp'
```

or through CMake (target and test `nn_mlp`).

## Files

| File | Contents |
|---|---|
| [tensor.hpp](tensor.hpp) | `Tensor<Extents>`, row-major, shaped by a `std::extents` whose extents are each static or dynamic (`nn::dyn`); `Vec<N>` and `Mat<R,C>` are aliases (`Mat<nn::dyn, 3>`: dynamic rows, 3 columns). Fully static shapes use `std::array` storage (trivially copyable, usable as `constexpr` constants); any dynamic extent uses `std::vector`. `view()` is a `std::mdspan`. Ops mix static and dynamic extents (`Vec<3> + Vec<nn::dyn>` is a `Vec<3>`). Static extents that disagree are a compile error, dynamic ones a `std::invalid_argument`. Primitives: `add sub hadamard scale relu sigmoid sum dot mse matvec matmul transpose outer`. Operators `* + -` as one-line wrappers. Helpers for rules and training (`zeros`, `zeros_like`, `like_as`, `relu_mask`, `sigmoid_slope`, `matvec_transposed`, `filled`, `filled_like`, `sgd_update`). No reflection. |
| [autodiff.hpp](autodiff.hpp) | The engine, which knows no tensor type and no primitive. `engine<Rules...>` reflects a body into a DAG, runs activity and liveness analysis, and its `value_and_grad` / `grad` generate the forward and reverse sweeps; `tape<Fn, Mask>` exposes the analysed DAG for inspection. `rule_set<...>` groups rules. |
| [rules.hpp](rules.hpp) | `nn::tensor_rule_set`: one rule class per primitive, in `nn::tensor_rules`, giving the vector-Jacobian product (VJP) per argument. |
| [cmath_rules.hpp](cmath_rules.hpp) | `nn::ad::cmath_rule_set`: `exp log sqrt sin cos tanh` as primitives (float, double, long double), each a class in `nn::ad::cmath_rules` giving dy/dx. |
| [nn.hpp](nn.hpp) | Includes all three; `default_rules` (tensor, then `<cmath>`) and `nn::ad::value_and_grad`, `grad` and `tape` with them. |

## How it works

1. **Lowering** (consteval, `detail::build`):
   - `statements_of(body_of(^^loss))` is walked. Each `T v = expr;` and the
     `return expr;` become SSA nodes.
   - A call becomes a node if one of the engine's rules lists its callee, or
     for a template specialization its template. The first such rule wins.
   - Any other callee is **inlined**: `body_of(callee)` is lowered into the
     same DAG, with its parameters bound to the argument slots. This is how
     `W * x + b` works: `operator*` / `operator+` have no rule, so their
     one-line bodies are inlined and expose `matvec` / `add`. User layers
     (`dense_relu`) work the same way, at any shape.
   - Built-in `+ - * /` and unary `-` on scalars are handled natively, so scalar
     C++ can surround the tensor code. Literals and constant globals are folded
     to constants; the XOR data in the test are `constexpr Vec`s.
2. **Types.** The proposal has no `type_of(expression)`. Node types come from
   three sources:
   - `return_type_of(callee)` for calls;
   - parameter and variable types for bindings;
   - `decltype` through an alias template (`substitute(^^mul_t, {A, B})`) for
     built-in operators.

   Arithmetic conversions hidden in casts are restored as `Convert` nodes where
   the target type is known.
3. **Analysis:**
   - *varied*: depends on one of the `Wrt` parameters, as in `requires_grad`;
   - *live*: reaches the result;
   - *needed*: the reverse sweep reads it, or a needed value is computed from
     it. A rule declares what its pullback reads,
     `static constexpr nn::ad::reads reads = nn::ad::reads::args;`
     (`result`, `args`, `nothing`, or the default `all`); built-in `*` and `/`
     read their operands.

   Dead nodes are never evaluated. Non-varied nodes get no reverse work. A rule
   is only asked for the adjoints of varied arguments. `value_and_grad`
   evaluates the live nodes; `grad` only the needed ones, so it skips the loss
   itself and anything only it depends on.
4. **Codegen:**
   - Values and adjoints live in a `std::tuple` of the node types. An
     adjoint's first contribution is assigned and later ones added, so no
     zero of a value's shape is needed (`grad` may not have the value). A
     parameter no contribution reaches gets `zeros_like(argument)`, found by
     ADL (or `T{}`).
   - Two expansion statements (`template for`) emit one statement per node.
     Forward: `std::get<i>(val) = [:fn:](args...)`. Reverse:
     `accumulate(adj[arg], Rule::pullback<k>(adj[i], val[i], args...))`.
   - There is no run-time tape, graph object or virtual call. Tensors with
     dynamic extents allocate their storage.

A rule looks like this:

```cpp
struct matvec {                                                // z = W x
  static constexpr std::span<const info> callees = listed({^^nn::matvec});
  static constexpr nn::ad::reads reads = nn::ad::reads::args;
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &z, const auto &W, const auto &x) {
    if constexpr (I == 0) return nn::outer(g, x);              // W̄ = z̄ xᵀ
    else                  return nn::matvec_transposed(W, g);  // x̄ = Wᵀ z̄
  }
};
```

The engine names no rule: `engine<Rules...>` takes rules and `rule_set`s of
them, so a scalar engine and a tensor engine are the same code with different
rule sets, and a rule listed first overrides a later one.

A primitive can have loops and anything else inside; only its rule matters.
Code calling primitives must be straight-line. That is the split between
"tensor level is straight-line, loops live inside primitives" and "every
primitive has a hand-written VJP", which is the same design as PyTorch / JAX
(Python libraries) at the op level.

A function that can't be named can't be listed directly: `^^std::exp` names
an overload set. Its rule lists the overloads it covers, recovered by
`probe_callee` from a one-line function calling them. A unary scalar rule can
give dy/dx instead of `pullback`, from the argument (`d_x`) or from the result
(`d_y`); which one also says what the rule reads.

```cpp
namespace nn::ad::cmath_probe {
template <class T> T exp(T x) { return std::exp(x); }
}
struct nn::ad::cmath_rules::exp {
  static constexpr std::span<const info> callees = probed(
      {^^cmath_probe::exp<float>, ^^cmath_probe::exp<double>, ^^cmath_probe::exp<long double>});
  static constexpr auto d_y(auto y) { return y; }   // (eˣ)' = eˣ = y
};
```

## Design notes

- **Feed-forward networks are straight-line at the tensor level.** For each
  operation, z = Wx gives W̄ += z̄ xᵀ and x̄ += Wᵀ z̄. A bias add gives
  b̄ += z̄, ReLU gives a mask, and MSE gives ȳ = 2(y − t)/n. The proposal's
  current surface (declarations, return, operators, calls, `callee_of`,
  `body_of`) is enough for this. The test demonstrates it.
- **Comparison with Python frameworks:**
  - PyTorch autograd is an operator-overloading tape that is rebuilt on every
    call; it is not tracing.
  - `torch.compile`/TorchDynamo analyses bytecode, with graph breaks.
  - JAX traces Python into jaxpr, and the functional style there is for purity.
    It does not mean "no loops": loop limits come from tracing, so you use
    `lax.scan` / `fori_loop`. JAX is not NN-only.
  - Tangent did source-to-source AD on Python ASTs; it is archived.
  - Reflection gets the structure at compile time from ordinary C++ functions.
    That needs no tracing, no graph rebuild and no graph breaks, and the DAG
    is fixed in the binary.
- **The engine gives no capability that PyTorch/JAX lack.** Their real
  advantages are the infrastructure: kernel libraries, fusion, distributed
  training, and GPU/TPU backends. This proof of concept (PoC) does not compete
  there; it shows the language can express the front end.
- **Proposal limits and the workarounds used here:**
  - *No control-flow traversal.* Depth is fixed by templates / `if constexpr`.
    The discarded branch is not returned, since reflection runs after
    instantiation. Loops go inside primitives.
  - *Casts are deferred*, so their target type is unknown. Expression-template
    libraries such as Eigen therefore don't fit: they rely on proxies and
    implicit conversions. The library uses value-returning ops only.
    An explicit cast is peeled like an implicit conversion, so a
    value-changing one (`static_cast<int>(x)`) is silently lost: Clang can't
    tell the two apart (GCC has `is_cast`), and a conversion's target type
    can't be queried.
  - *No in-place ops in reflected code.* Write `T v = expr;`, not `+=`.
  - *No `type_of(expr)`.* See "Types" above.
  - *No expression splicing / code injection.* Codegen uses the existing
    splices (`[:fn:]`, `[:value:]`) plus `template for` over the DAG.
- **Proof-of-concept milestones:**
  1. dense + MSE, checked against finite differences — **done**;
  2. 2-layer MLP on XOR — **done**;
  3. depth fixed by a template;
  4. NN mixed with scalar C++ such as Black-Scholes or physics. The scalar
     glue works (`regularised` in the test), and `<cmath>` calls are
     primitives (`cmath_mix`); a model of that kind is still to do.

## Compiler notes

The engine goes through `ad::meta` (`include/ad/meta.h`), which is plain
`std::meta`: both forks have the same statement and expression API.

Both forks:
- `operands_of(call)` is the arguments only; the callee comes from
  `declaration_of(callee_of(call))`.
- An id-expression is an expression, not the entity it names: `constant_of`,
  `identifier_of`, `type_of` and the other entity queries throw on it, so the
  engine asks them of `declaration_of(e)`.
- `type_of` on an expression is an error.
- An **overloaded operator is both `is_function_call` and
  `is_operator_expression`**, so the engine checks for calls first. The
  proposal (§4.2) says `a+b` classifies as an operator.
- Operands come in the order written (`0.5 * s` is `0.5, s`).
- A copy of a class value is a conversion.

Left to the implementation (the proposal doesn't specify which implicit
conversions are shown), so the engine peels any conversion it meets and
restores arithmetic ones itself:
- Clang shows reading a scalar variable as a conversion (lvalue-to-rvalue),
  and `double four = 4;` as a conversion of the literal `4`.
- GCC shows neither: the variable is read directly, and `double four = 4;` is
  the literal `4.0`.

GCC (a gap in its P2996 support, not in P4395):
- **Registry identity bug.** `template_of(spec) == ^^nn::matvec` is true, yet
  GCC treats the two as different template arguments: comparing
  `X<template_of(spec)>` with `X<^^nn::matvec>` causes an internal
  compiler error (ICE), "canonical types differ for identical types"
  (`tests/known_gap/gcc/template_of_as_template_argument.cpp`, #272). The
  engine only compares reflections with `==`, never uses them as template
  arguments, so it isn't affected.
- **A range-for over `members_of(namespace)` sees the namespace as it was
  where the function is defined**. The engine takes its rules as template
  arguments, so it doesn't scan namespaces.

Clang accepts more than the standard here: `extract<T>` by value goes through
`constant_of`, i.e. `reflect_constant`, so `T` must be structural. GCC rejects
a `std::span` and Clang doesn't; the engine reads `callees` as
`extract<const std::span<const info> &>`, valid on both.

## GPU: possible directions (not implemented; current builds are CPU-only)

This repo's builds have no GPU target: the Clang fork is built without
NVPTX and GCC without offload targets.

1. **Host-side reflection, library kernels (works with both compilers).**
   The engine already treats primitives as opaque calls. Replace a primitive's
   body (and its rule's helpers) with cuBLAS / cuDNN calls on
   device-resident tensors. Reflection and codegen stay in host code, so this
   works with either fork. It needs Linux plus CUDA, and a tensor whose
   storage is a device buffer; primitives can pass kernels its `mdspan`.
2. **Custom kernels in separate `.cu` files** compiled by nvcc, called from
   primitives. Same as (1).
3. **Device code generated from reflection** (generated fused kernels) needs
   a compiler that both reflects and targets the GPU:
   - **Clang:** Clang compiles CUDA natively. Rebuild the fork on Linux with
     `LLVM_TARGETS_TO_BUILD="AArch64;NVPTX"` (or X86;NVPTX) plus the CUDA SDK,
     then reflection and `__global__` code share one front end. This is the
     most promising route.
   - **GCC:** GCC cannot compile CUDA. It can be nvcc's *host* compiler, but
     nvcc's EDG front end (cudafe++) parses the `.cu` file first and does not
     know `^^` / `template for`. So reflection must stay in `.cpp` files that
     GCC compiles, with kernels in `.cu`. GCC's own offload is OpenMP/OpenACC
     with `-foffload=nvptx-none` (or amdgcn), configured with
     `--enable-offload-targets`, on Linux. Generated code could put
     `#pragma omp target` loops inside primitives. *Unverified:* whether
     reflection-generated code survives into the offload compilation (the
     "lowered IR goes to the offload compiler" assumption).
4. Fusion at reflection time is the interesting research angle. The whole
   DAG is known at compile time, so adjacent elementwise primitives
   (`add` → `relu`) could be merged into one generated loop or kernel. That is
   what `torch.compile` does at run time.

## Status and next steps

Done:
- the library;
- [tests/common/nn_mlp.t.cpp](../../tests/common/nn_mlp.t.cpp), passing on
  Clang and GCC via `run_tests.py` and via CMake (Clang RelWithDebInfo; GCC
  Debug + sanitizers);
- mutation-checked: breaking the sigmoid rule, or replacing accumulation with
  assignment, fails the test.

Next, roughly in order:
1. **Template-depth MLP** (milestone 3), e.g. `mlp<Sizes...>` recursing
   through a layer template with `if constexpr`. Check how discarded branches
   reflect on both compilers.
2. **A Black-Scholes-style model with a NN term** (milestone 4), using the
   `<cmath>` primitives. Integer-argument overloads (`std::exp(1)`) and
   binary functions (`pow`, `atan2`, which need `pullback`) are not covered.
3. **Batched data**: `Mat<B, In>` inputs with a row-broadcast bias primitive,
   instead of one inlined copy per sample.
4. More ops: elementwise `tanh` on tensors, softmax + cross-entropy (one fused primitive, for
   stability), maybe convolution as a primitive.
5. GPU route 1 on a Linux + CUDA machine, then the Clang+NVPTX rebuild.
