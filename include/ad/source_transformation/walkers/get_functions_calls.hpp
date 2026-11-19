#pragma once

#include "../../autograd.h"
#include "../../meta.h"
#include "../throw_helper.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;
namespace su = ad::string_util;

struct GetFunctionCalls {
  // TODO: Hack to get function calls since meta::info objects can not be used in constexpr context.
  static consteval void run(info o, GenerationContext& ctx) {
    info obj = ad::detail::stripCasts(o); // TODO: Handle casts.

    if (m::is_operator_expression(obj)) {
      auto operands = operands_of(obj);

      for (info operand : operands) {
        run(operand, ctx);
      }
    } else if (m::is_literal(obj)) {
      // Nothing
    } else if (m::is_id_expression(obj)) {
      // Nothing
    } else if (m::is_function_call(obj)) {
      // First handle arguments
      auto operands = operands_of(obj);
      for (info operand : operands) {
        run(operand, ctx);
      }

      // Add function call to ctx
      info decl = m::declaration_of(m::callee_of(obj));
      ctx.addRequiredFunction(decl);
    } else if (m::is_declaration_statement(obj)) {
      info v = declared_variable_of(obj);
      run(initializer_of(v), ctx);
    } else if (m::is_expression_statement(obj)) {
      run(expression_of(obj), ctx);
    } else if (m::is_return_statement(obj)) {
      run(return_operand_of(obj), ctx);
    } else {
      throwKind(obj);
    }
  }
};

}