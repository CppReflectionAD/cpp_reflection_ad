// cx_fp_contract.hpp — turn off FMA contraction for the cx headers.
//
// The error-free transformations in the cx headers (Veltkamp's split,
// Dekker's product, the Cody–Waite reduction) need every product rounded on
// its own. Fusing a product into the following add or subtract (FMA
// contraction, which compilers do by default on targets with FMA, such as
// arm64) breaks them, so each header wraps its own code in
// CX_FP_CONTRACT_OFF_BEGIN / CX_FP_CONTRACT_OFF_END. Code that includes the
// headers keeps its own contraction settings.

#ifndef CX_STD_CX_FP_CONTRACT_HPP
#define CX_STD_CX_FP_CONTRACT_HPP

#if defined(__clang__)
#define CX_FP_CONTRACT_OFF_BEGIN                                               \
  _Pragma("float_control(push)") _Pragma("clang fp contract(off)")
#define CX_FP_CONTRACT_OFF_END _Pragma("float_control(pop)")
#elif defined(__GNUC__)
#define CX_FP_CONTRACT_OFF_BEGIN                                               \
  _Pragma("GCC push_options") _Pragma("GCC optimize(\"fp-contract=off\")")
#define CX_FP_CONTRACT_OFF_END _Pragma("GCC pop_options")
#else
#define CX_FP_CONTRACT_OFF_BEGIN
#define CX_FP_CONTRACT_OFF_END
#endif

#endif // CX_STD_CX_FP_CONTRACT_HPP
