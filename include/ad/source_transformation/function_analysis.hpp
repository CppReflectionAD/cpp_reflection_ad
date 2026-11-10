#pragma once

#include <sstream>
#include <fstream>
#include <ranges>
#include <iostream>

#include "../autograd.h"
#include "../meta.h"
#include "generation_context.hpp"
#include "operator_info.hpp"
#include "string_util.hpp"
#include "throw_helper.hpp"

#include "ast/meta_ast.hpp"
#include "walkers/extract_function_walker.hpp"
#include "walkers/forward_ad_walker.hpp"
#include "walkers/get_functions_calls.hpp"
#include "walkers/reverse_ad_walker.hpp"
#include "walkers/to_cpp_walker.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;
namespace su = ad::string_util;

struct FunctionAnalysis {
  static consteval auto generatePrimal(info func) {
    GenerationContext ctx = {};
    std::string r = "";
    info ret_type = m::return_type_of(func);
    auto params = m::parameters_of(func);

    std::string params_s = "";
    for (info p : params) {
      params_s += su::join(params_s.size() ? ", " : "", m::display_string_of(m::type_of(p)), " ", m::identifier_of(p));
    }

    r += su::join(m::display_string_of(ret_type), " ", m::identifier_of(func), "(", params_s, ") {\n");

    std::vector<Statement*> stmts = fromFunctionBody(m::body_of(func));
    ToCppWalker walker = {&ctx};
    ctx.addIndent();

    walker.walk(&stmts);
    r += walker.code;

    ctx.removeIndent();
    r += "}\n";

    releaseFunctionBody(stmts);

    return std::define_static_string(r);
  }

  static consteval auto generateForward(info func, GenerationContext& ctx) {
    std::string r = "";
    info ret_type = m::return_type_of(func);
    auto params = m::parameters_of(func);

    std::string params_s = "";
    for (info p : params) {
      params_s += su::join(
        params_s.size() ? ", " : "",
        m::display_string_of(m::type_of(p)), " ", m::identifier_of(p), ", ",
        m::display_string_of(m::type_of(p)), " ", m::identifier_of(p), "_d");
    }

    r += su::join("std::pair<", m::display_string_of(ret_type), ", ", m::display_string_of(ret_type), "> ", m::identifier_of(func), "_d(", params_s, ") {\n");

    std::vector<Statement*> stmts = fromFunctionBody(m::body_of(func));

    // Replace function calls with temporaries.
    ExtractFunctionWalker extFnc = {&ctx};
    extFnc.walk(&stmts);

    ForwardADWalker ad = {&ctx};
    ad.walk(&stmts);

    ctx.addIndent();
    ToCppWalker walker = {&ctx};
    walker.walk(&stmts);
    r += walker.code;
    ctx.removeIndent();

    r += "}\n";

    releaseFunctionBody(stmts);

    // TODO: Remove when function declarations are part of the AST.
    for (info stmt : m::statements_of(m::body_of(func))) {
      GetFunctionCalls::run(stmt, ctx);
    }

    return std::define_static_string(r);
  }

  static consteval auto generateReverseJoined(info func, GenerationContext& ctx) {
    std::string r = "";
    info ret_type = m::return_type_of(func);
    auto params = m::parameters_of(func);

    std::string params_s = "";
    for (info p : params) {
      params_s += su::join(
        params_s.size() ? ", " : "",
        m::display_string_of(m::type_of(p)), " ", m::identifier_of(p), ", ",
        m::display_string_of(m::type_of(p)), "& ", m::identifier_of(p), "_b");
    }
    if (!m::is_void_type(ret_type)) {
      params_s += su::join(
        params_s.size() ? ", " : "",
        m::display_string_of(ret_type), " ret_b");
    }

    // Reverse functions do not return values.
    r += su::join("void ", m::identifier_of(func), "_b(", params_s, ") {\n");

    std::vector<Statement*> stmts = fromFunctionBody(m::body_of(func));

    // Replace function calls with temporaries.
    ExtractFunctionWalker extFnc = {&ctx};
    extFnc.walk(&stmts);

    ReverseADWalker ad = {&ctx};
    ad.walk(&stmts);
    std::reverse(ad.reverse_statements.begin(), ad.reverse_statements.end());

    ctx.addIndent();
    ToCppWalker walker = {&ctx};
    walker.walk(&stmts);
    walker.walk(&ad.reverse_statements);
    r += walker.code;
    ctx.removeIndent();

    r += "}\n";

    releaseFunctionBody(stmts);
    releaseFunctionBody(ad.reverse_statements);

    // TODO: Remove when function declarations are part of the AST.
    for (info stmt : m::statements_of(m::body_of(func))) {
      GetFunctionCalls::run(stmt, ctx);
    }

    // ctx.addIndent();
    // for (info stmt : m::statements_of(m::body_of(func))) {
    //   bool is_return = m::is_return_statement(stmt);
    //
    //   StatementAnalysis::generateStore(stmt, ctx);
    //   StatementAnalysis::generateReverseDecl(stmt, ctx);
    //   StatementAnalysis::generateReverse(stmt, ctx, !is_return);
    //   StatementAnalysis::generateRestore(stmt, ctx);
    //   ctx.addReverseStatement(su::join("//", StatementAnalysis::generatePrimal(stmt, ctx)));
    //   ctx.addReverseStatement("");
    // }
    //
    // r += ctx.flushStatements();
    // r += ctx.flushReverseStatements();
    //
    // ctx.removeIndent();
    // r += "}\n";

    return std::define_static_string(r);
  }
};
}

