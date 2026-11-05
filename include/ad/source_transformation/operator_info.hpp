#pragma once

#include "../meta.h"

namespace ad {
using std::meta::info;
namespace m = ad::meta;

enum class OperatorKind {
  Assign,
  Mul
};

struct OperatorInfo {
  OperatorKind kind;
  std::string symbol;
};

constexpr OperatorInfo getOperatorInfo(m::operators op) {
  if (op == m::operators::op_equals) {
    return {OperatorKind::Assign, "="};
  } else if (op == m::operators::op_star) {
    return {OperatorKind::Mul, "*"};
  } else {
    throw "Unknown operator";
  }
}
}