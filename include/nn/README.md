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
| [tensor.hpp](tensor.hpp) | `Vec<N>`, `Mat<R,C>` (fixed size, `std::array` storage, trivially copyable). Primitives: `add sub hadamard scale relu sigmoid sum dot mse matvec matmul transpose outer`. Operators `* + -` as one-line wrappers. Helpers for rules and training (`relu_mask`, `sigmoid_slope`, `matvec_transposed`, `filled`, `sgd_update`). No reflection. |
| [autodiff.hpp](autodiff.hpp) | The engine, which knows no tensor type. It reflects a body into a DAG, then runs activity and liveness analysis. `value_and_grad` / `grad` generate the forward and reverse sweeps. `tape<Fn, Mask>` exposes the analysed DAG for inspection. |
| [rules.hpp](rules.hpp) | One `nn::ad::vjp<key(^^nn::f)>` per primitive: the vector-Jacobian product (VJP) per argument. |
| [nn.hpp](nn.hpp) | Includes all three. |

## How it works

1. **Lowering** (consteval, `detail::build`):
   - `statements_of(body_of(^^loss))` is walked. Each `T v = expr;` and the
     `return expr;` become SSA nodes.
   - A call becomes a node if its callee has a rule. The rule is looked up by
     reflection: `vjp<key(callee)>`, where a template specialization's key is
     its template.
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
   - *live*: reaches the result.

   Dead nodes are never evaluated. Non-varied nodes get no reverse work. A rule
   is only asked for the adjoints of varied arguments.
4. **Codegen:**
   - Values and adjoints live in a `std::tuple` of the node types, with the
     adjoints zero-initialized.
   - Two expansion statements (`template for`) emit one statement per node.
     Forward: `std::get<i>(val) = [:fn:](args...)`. Reverse:
     `accumulate(adj[arg], Rule::pullback<k>(adj[i], val[i], args...))`.
   - There is no run-time tape, graph object, virtual call or allocation.

A rule looks like this:

```cpp
template <> struct nn::ad::vjp<nn::ad::key(^^nn::matvec)> {   // z = W x
  template <std::size_t I>
  static constexpr auto pullback(const auto &g, const auto &z, const auto &W, const auto &x) {
    if constexpr (I == 0) return nn::outer(g, x);              // W̄ = z̄ xᵀ
    else                  return nn::matvec_transposed(W, g);  // x̄ = Wᵀ z̄
  }
};
```

A primitive can have loops and anything else inside; only its rule matters.
Code calling primitives must be straight-line. That is the split between
"tensor level is straight-line, loops live inside primitives" and "every
primitive has a hand-written VJP", which is the same design as PyTorch / JAX
(Python libraries) at the op level.

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
  - *No in-place ops in reflected code.* Write `T v = expr;`, not `+=`.
  - *No `type_of(expr)`.* See "Types" above.
  - *No expression splicing / code injection.* Codegen uses the existing
    splices (`[:fn:]`, `[:value:]`) plus `template for` over the DAG.
- **Proof-of-concept milestones:**
  1. dense + MSE, checked against finite differences — **done**;
  2. 2-layer MLP on XOR — **done**;
  3. depth fixed by a template;
  4. NN mixed with scalar C++ such as Black-Scholes or physics. The scalar
     glue works (`regularised` in the test), but scalar `<cmath>` calls are not
     primitives here yet; see next steps.

## Compiler notes

The engine goes through `ad::meta` (`include/ad/meta.h`), which gives the
clang fork's expression API on both compilers. With it, on both forks:
- `operands_of(call)` is the arguments only; the callee comes from
  `declaration_of(callee_of(call))`.
- `type_of` on an expression is an error.

Clang:
- An **overloaded operator is both `is_function_call` and
  `is_operator_expression`**, so the engine checks for calls first. The
  proposal (§4.2) says `a+b` classifies as an operator.
- Copies are one-operand `construct`s.

GCC:
- An overloaded operator is `is_function_call` only.
- A class with a non-trivial destructor used to lose the `return` statement
  from `statements_of`. Fixed in the pinned fork; see
  [tests/common/nontrivial_destructor_reflection.cpp](../../tests/common/nontrivial_destructor_reflection.cpp).
  A temporary of such a class is a three-operand construct: the unnamed
  variable, its initializer, and the destructor call that cleans it up.
  `strip()` peels it.
- **A second local of such a class has no initializer** (`initializer_of`
  throws), so a reflected body can name only one intermediate of a
  heap-backed type. See
  [tests/known_gap/gcc/initializer_after_cleanup.cpp](../../tests/known_gap/gcc/initializer_after_cleanup.cpp).
- Binding to `const T&` shows as `cast(&x)`. A temporary or return slot is
  `construct(<unnamed var>, init)`. `strip()` in autodiff.hpp peels both.
- **Each declaration statement appears twice** in `statements_of`. The engine
  de-duplicates by name; a block cannot declare a name twice.
- Commutative built-in operands may be swapped (`0.5 * s` → `s, 0.5`), and
  `double four = 4;` is folded to `4.0` (no conversion node).
- **Registry identity bug.** `template_of(spec) == ^^nn::matvec` is true, yet
  `substitute(^^vjp, {reflect_constant(template_of(spec))})` does not find
  `vjp<^^nn::matvec>`. Comparing the two specializations even causes an
  internal compiler error (ICE): "canonical types differ for identical types".
  The workaround is `nn::ad::key()`, which normalises via `members_of(parent)`
  and must be used when registering.
- `expression_of` / `is_expression_statement` are absent.

## GPU: possible directions (not implemented; current builds are CPU-only)

The current builds target macOS AArch64. The Clang fork is built with
`LLVM_TARGETS_TO_BUILD=AArch64` (no NVPTX). GCC is configured for
aarch64-apple-darwin without offload targets. macOS has no CUDA.

1. **Host-side reflection, library kernels (works with both compilers).**
   The engine already treats primitives as opaque calls. Replace a primitive's
   body (and its rule's helpers) with cuBLAS / cuDNN calls on
   device-resident tensors. Reflection and codegen stay in host code, so this
   works with either fork. It needs Linux plus CUDA, and a tensor type holding
   a device pointer. That type must be trivially copyable for GCC; a non-owning
   handle is.
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
2. **Scalar `<cmath>` primitives** (`exp`, `log`, `tanh`, ...) for mixing with
   scalar models. Reuse the probe trick from `include/ad/autograd.h`:
   `^^std::exp` names an overload set, so recover the callee from a one-line
   probe function. Then a Black-Scholes-style model with a NN term
   (milestone 4).
3. **Batched data**: `Mat<B, In>` inputs with a row-broadcast bias primitive,
   instead of one inlined copy per sample.
4. More ops: `tanh`, softmax + cross-entropy (one fused primitive, for
   stability), maybe convolution as a primitive.
5. A `known_gap/gcc` test for the registry-identity bug (see compiler notes).
6. GPU route 1 on a Linux + CUDA machine, then the Clang+NVPTX rebuild.
