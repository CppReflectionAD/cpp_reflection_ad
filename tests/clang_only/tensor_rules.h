// tensor_rules.h — registers nn's op vocabulary with the AD engine.
//
// tensor.h stays AD-agnostic; this header is the seam between the two. The
// engine keys primitives on the callee reflection, so these declare "nn::relu
// IS the Relu primitive" — a `relu` in any other namespace is unaffected and
// would be inlined instead.
//
// Registration is required for the tensor ops: their bodies are loops, not
// straight-line scalar code, so inlining them is not possible — the engine's
// hand-written VJPs are what make them differentiable.

#ifndef REFLECT_DEMO_TENSOR_RULES_H
#define REFLECT_DEMO_TENSOR_RULES_H

#include "../autograd.h"
#include "tensor.h"

template <> struct ad::primitive<^^nn::add>       { static constexpr ad::OpKind op = ad::OpKind::Add; };
template <> struct ad::primitive<^^nn::sub>       { static constexpr ad::OpKind op = ad::OpKind::Sub; };
template <> struct ad::primitive<^^nn::mul>       { static constexpr ad::OpKind op = ad::OpKind::Mul; };
template <> struct ad::primitive<^^nn::div>       { static constexpr ad::OpKind op = ad::OpKind::Div; };
template <> struct ad::primitive<^^nn::matmul>    { static constexpr ad::OpKind op = ad::OpKind::Matmul; };
template <> struct ad::primitive<^^nn::transpose> { static constexpr ad::OpKind op = ad::OpKind::Transpose; };
template <> struct ad::primitive<^^nn::sum>       { static constexpr ad::OpKind op = ad::OpKind::Sum; };
template <> struct ad::primitive<^^nn::relu>      { static constexpr ad::OpKind op = ad::OpKind::Relu; };

#endif  // REFLECT_DEMO_TENSOR_RULES_H
