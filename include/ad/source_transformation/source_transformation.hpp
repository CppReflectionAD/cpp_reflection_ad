#pragma once

#include "../meta.h"

#include "function_analysis.hpp"

namespace ad {
using std::meta::info;
namespace m = ad::meta;
namespace su = ad::string_util;

struct Ignore {};

#if __has_feature(annotation_attributes)
consteval bool hasIgnore(info o) {
  return 0 != std::meta::annotations_of_with_type(o, ^^ad::Ignore).size();
}
#endif


template<info o>
consteval auto handleNamespacePrimal() {
  static constexpr auto members = std::define_static_array(ad::m::members_of( o, ad::m::access_context::unchecked() ));

  std::string r = {};
  template for (constexpr auto member : members) {
    if (is_function(member)) { // TODO: Add ignore when attributes are available
      r += ad::FunctionAnalysis::generatePrimal(member);
      r += "\n";
    }
  }

  return std::define_static_string(r);
}

consteval std::string addRequiredFunctionsForward(GenerationContext& ctx) {

  std::string r = {};

  std::vector<info> functions = ctx.getRequiredFunctions();
  for (info func : functions) {
    r +=  ad::FunctionAnalysis::generateForward(func, ctx);
    r += "\n";
  }

  return r;
}

template<info o>
consteval auto handleNamespaceForward() {
  static constexpr auto members = std::define_static_array(ad::m::members_of( o, ad::m::access_context::unchecked() ));
  GenerationContext ctx = {};

  std::string r = {};

  template for (constexpr auto member : members) {
    if (is_function(member)) { // TODO: Add ignore when attributes are available
      std::string func = ad::FunctionAnalysis::generateForward(member, ctx);

      // Add required functions bevor the function is added.
      r += addRequiredFunctionsForward(ctx);

      r += func + "\n";
    }
  }

  return std::define_static_string(r);
}

consteval std::string addRequiredFunctionsReverse(GenerationContext& ctx) {
  std::string r = {};

  std::vector<info> functions = ctx.getRequiredFunctions();
  for (info func : functions) {
    r += ad::FunctionAnalysis::generateReverseJoined(func, ctx); // TODO: Handle split functions
    r += "\n";
  }

  return r;
}

template<info o>
consteval auto handleNamespaceReverse() {
  static constexpr auto members = std::define_static_array(ad::m::members_of( o, ad::m::access_context::unchecked() ));
  GenerationContext ctx = {};

  std::string r = {};
  template for (constexpr auto member : members) {
    if (is_function(member)) { // TODO: Add ignore when attributes are available
      std::string func = ad::FunctionAnalysis::generateReverseJoined(member, ctx);

      // Add required functions bevor the function is added.
      r += addRequiredFunctionsReverse(ctx);

      r += func + "\n";
    }
  }

  return std::define_static_string(r);
}

}

