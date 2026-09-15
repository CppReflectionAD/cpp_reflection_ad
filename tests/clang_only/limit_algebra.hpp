#ifndef LIMIT_ALGEBRA_HPP
#define LIMIT_ALGEBRA_HPP

// limit_algebra.hpp — compile-time limits by composition, not by search.
//
// Walks the same SSA/DAG autograd.h builds, carrying one *limit value* per
// node and combining them with a per-op rule. There is no neighbourhood, no
// shrinking sweep and no tolerance: exp(-x) as x -> +inf is 0 because the rule
// for exp says so, and the sigmoid built on it is 1 because the rules for +
// and / say so. One pass, and the answer is exact.
//
// Usage:
//   static_assert(ad::limits::is_convergent_at<^^poly>(ad::limits::At{2.0}));
//   constexpr double l = ad::limits::limit_of<^^poly>(ad::limits::At{2.0});
//
//   using ad::limits::At;
//   At{2.0}              // x -> 2, two-sided
//   At::from_left(2.0)   // x -> 2-
//   At::plus_infinity()  // x -> +inf
//
// A limit value is an extended real *plus the side it is approached from*:
//
//   Finite(v, Below|Exactly|Above) | PlusInf | MinusInf | None
//
// The side is not decoration. 1/u depends on whether u is 0- or 0+; exp(-x)
// tending to 0 is useless downstream unless it is known to be 0+; and a
// comparison at the switching point (`x < 1` as x -> 1-) is decidable only
// from the side. `None` is the bottom element: no limit, or one this algebra
// cannot name. It is not an error -- it flows, so a domain error on the dead
// side of a branch costs nothing, which is why (unlike the interval engine)
// there is no separate poison flag to keep it from laundering through min/max.
//
// Two-sided limits are two one-sided walks that must agree. That is the whole
// trick: on a one-sided walk every comparison at the limit point is decided
// (`x < 1` is true as x -> 1-, false as x -> 1+), so a branch is never
// "undecided over a box that straddles the seam". A jump is then reported
// because the sides disagree (fn_step: 0 vs 1), and a seam that happens to be
// continuous is *proved*, not missed (staircase at 1: 1 from both sides).
//
// Sound but incomplete, and the incompleteness is now exactly one thing: the
// indeterminate forms. inf-inf, 0*inf, inf/inf and 0/0 have no rule, so
// x - x, x/x and x*x/x out at infinity report None -- as does anything whose
// limit exists only by a cancellation the algebra cannot see. That is a
// deliberate starting point, not an oversight: an L'Hopital or series-order
// extension slots in as a richer element, not as a rewrite.
//
// Known holes:
//   * sin/cos have no constexpr kernel (see cx_std), so a finite limit point
//     is None rather than sin(a); at ±inf they are None for the right reason.
//   * Values are computed in double, so a finite limit is exact only to the
//     rounding of the ops on the path.
//   * Overflow (exp(1000)) reports PlusInf rather than a finite value double
//     cannot hold.

#include "../autograd.h" // ad::Node, ad::OpKind, ad::build_nodes<>
#include "../cx_std/cx_erfc.hpp"
#include "../cx_std/cx_exp.hpp"
#include "../cx_std/cx_log.hpp"
#include "../cx_std/cx_sqrt.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace ad::limits {

// ---------------------------------------------------------------------------
// The lattice element
// ---------------------------------------------------------------------------

// Which side the values come from: f - L is negative, zero, positive, or not
// determined.
enum class Side { Below, Exactly, Above, Unknown };

struct Value {
  enum class Kind { Finite, PlusInf, MinusInf, None };

  Kind kind = Kind::None;
  double value = 0.0; // Finite only
  Side side = Side::Unknown;

  constexpr bool finite() const { return kind == Kind::Finite; }
  constexpr bool NaN() const { return kind == Kind::None; }
  constexpr bool infinite() const {
    return kind == Kind::PlusInf || kind == Kind::MinusInf;
  }
};

// Where the moving operand is headed.
struct At {
  enum class Dir { Both, FromLeft, FromRight, PlusInf, MinusInf };

  double at = 0.0;
  Dir dir = Dir::Both;

  static consteval At from_left(double a) { return {a, Dir::FromLeft}; }
  static consteval At from_right(double a) { return {a, Dir::FromRight}; }
  static consteval At plus_infinity() { return {0.0, Dir::PlusInf}; }
  static consteval At minus_infinity() { return {0.0, Dir::MinusInf}; }
};

struct LimitResult {
  // A finite limit exists. False for a divergence to ±inf too -- read `limit`
  // to tell "diverges to +inf" from "no limit at all".
  bool converges = false;
  double value = 0.0; // NaN unless converges
  Value limit{};
  // The node that first produced None, and its op (-1 / Input if none did).
  int failing_node = -1;
  OpKind failing_op = OpKind::Input;
};

namespace detail_lim {

// --- infinitesimal sign algebra --------------------------------------------
//
// A side is the sign of f - L near the point, so the rules for combining sides
// are just sign arithmetic on infinitesimals. Kept as an int so they compose.
inline constexpr int kUnknownSign = 2;

constexpr int sign_of(Side s) {
  switch (s) {
  case Side::Below:
    return -1;
  case Side::Exactly:
    return 0;
  case Side::Above:
    return 1;
  case Side::Unknown:
    return kUnknownSign;
  }
  return kUnknownSign;
}

constexpr Side side_of(int sgn) {
  if (sgn == -1)
    return Side::Below;
  if (sgn == 0)
    return Side::Exactly;
  if (sgn == 1)
    return Side::Above;
  return Side::Unknown;
}

constexpr bool known(int sgn) { return sgn != kUnknownSign; }

// Sign of the sum of two infinitesimals: decided when they agree or one of
// them vanishes, otherwise a race this algebra does not run.
constexpr int sign_add(int p, int q) {
  if (p == 0)
    return q;
  if (q == 0)
    return p;
  if (!known(p) || !known(q))
    return kUnknownSign;
  return p == q ? p : kUnknownSign;
}

constexpr int sign_mul(int p, int q) {
  if (p == 0 || q == 0)
    return 0; // a vanishing factor wins even against an unknown sign
  if (!known(p) || !known(q))
    return kUnknownSign;
  return p * q;
}

constexpr int sign_neg(int p) { return known(p) ? -p : kUnknownSign; }

constexpr int sign_num(double x) { return x > 0.0 ? 1 : (x < 0.0 ? -1 : 0); }

// --- constructors ----------------------------------------------------------

constexpr double nan() { return std::numeric_limits<double>::quiet_NaN(); }

constexpr bool is_double_finite(double x) {
  return x == x && x != std::numeric_limits<double>::infinity() &&
         x != -std::numeric_limits<double>::infinity();
}

constexpr Value NaN() { return {Value::Kind::None, 0.0, Side::Unknown}; }
constexpr Value plus_inf() { return {Value::Kind::PlusInf, 0.0, Side::Below}; }
constexpr Value minus_inf() {
  return {Value::Kind::MinusInf, 0.0, Side::Above};
}

constexpr Value finite(double v, int sgn) {
  // An op that overflowed has no finite limit to report; ±inf is the honest
  // answer, and the caller learns the difference from `limit.kind`.
  if (!is_double_finite(v))
    return v > 0.0 ? plus_inf() : (v < 0.0 ? minus_inf() : NaN());
  return {Value::Kind::Finite, v, side_of(sgn)};
}

constexpr Value finite(double v, Side s) { return finite(v, sign_of(s)); }

// The sign f itself eventually has, as opposed to the sign of f - L: for a
// nonzero limit it is the limit's sign, and at zero it is the side.
constexpr int eventual_sign(const Value &u) {
  if (u.kind == Value::Kind::PlusInf)
    return 1;
  if (u.kind == Value::Kind::MinusInf)
    return -1;
  if (!u.finite())
    // NaN
    return kUnknownSign;
  const int s = sign_num(u.value);
  return s != 0 ? s : sign_of(u.side);
}

// --- arithmetic ------------------------------------------------------------

constexpr Value neg(const Value &u) {
  switch (u.kind) {
  case Value::Kind::None:
    return NaN();
  case Value::Kind::PlusInf:
    return minus_inf();
  case Value::Kind::MinusInf:
    return plus_inf();
  case Value::Kind::Finite:
    return finite(-u.value, sign_neg(sign_of(u.side)));
  }
  return NaN();
}

constexpr Value add(const Value &u, const Value &v) {
  if (u.NaN() || v.NaN())
    return NaN();

  if (u.infinite() || v.infinite()) {
    // inf - inf is undefined
    if (u.kind == Value::Kind::PlusInf && v.kind == Value::Kind::MinusInf)
      return NaN();
    if (u.kind == Value::Kind::MinusInf && v.kind == Value::Kind::PlusInf)
      return NaN();
    return u.infinite() ? Value{u.kind, 0.0, u.side}
                        : Value{v.kind, 0.0, v.side};
  }

  return finite(u.value + v.value, sign_add(sign_of(u.side), sign_of(v.side)));
}

constexpr Value sub(const Value &u, const Value &v) { return add(u, neg(v)); }

constexpr Value mul(const Value &u, const Value &v) {
  if (u.NaN() || v.NaN())
    return NaN();

  if (u.infinite() || v.infinite()) {
    // 0 * inf is undefined
    if ((u.finite() && u.value == 0.0) || (v.finite() && v.value == 0.0))
      return NaN();
    const int su = eventual_sign(u);
    const int sv = eventual_sign(v);
    if (!known(su) || !known(sv))
      return NaN();
    return su * sv > 0 ? plus_inf() : minus_inf();
  }

  const int pu = sign_of(u.side);
  const int pv = sign_of(v.side);
  // d(uv) = v du + u dv + du dv. The first-order terms decide it unless both
  // limits are zero, where the product of the two infinitesimals is all there
  // is -- and that is what makes (0-)(0-) approach 0 from above.
  int sgn = 0;
  if (u.value == 0.0 && v.value == 0.0) {
    sgn = sign_mul(pu, pv);
  } else {
    sgn = sign_add(sign_mul(sign_num(v.value), pu),
                   sign_mul(sign_num(u.value), pv));
  }
  return finite(u.value * v.value, sgn);
}

constexpr Value div(const Value &u, const Value &v) {
  if (u.NaN() || v.NaN())
    return NaN();

  if (v.infinite()) {
    if (u.infinite())
      return NaN(); // inf / inf
    // A finite over an infinity is 0, and the side is what the next op needs.
    return finite(0.0, sign_mul(eventual_sign(u),
                                v.kind == Value::Kind::PlusInf ? 1 : -1));
  }

  if (v.value == 0.0) {
    // 0 / 0, on the limits rather than on the nearby values: x²/x has a
    // numerator tending to 0+ and a denominator tending to 0-, and answering
    // -inf there would be wrong, not merely imprecise.
    if (u.finite() && u.value == 0.0)
      return NaN();
    const int sv = sign_of(v.side);
    // v is identically 0 near the point, or approaches from an undecided side:
    // either way there is no signed infinity to report.
    if (sv == 0 || !known(sv))
      return NaN();
    const int su = eventual_sign(u);
    if (!known(su))
      return NaN();
    return su * sv > 0 ? plus_inf() : minus_inf();
  }

  if (u.infinite()) {
    const int sv = sign_num(v.value);
    const int su = u.kind == Value::Kind::PlusInf ? 1 : -1;
    return su * sv > 0 ? plus_inf() : minus_inf();
  }

  // d(u/v) = (v du - u dv) / v², and v² > 0 does not touch the sign.
  const int sgn = sign_add(sign_mul(sign_num(v.value), sign_of(u.side)),
                           sign_mul(-sign_num(u.value), sign_of(v.side)));
  return finite(u.value / v.value, sgn);
}

// --- the smooth vocabulary -------------------------------------------------
//
// Each is monotone where it is defined, so a finite limit point is just
// "evaluate, and keep or flip the side". Only the endpoints need a rule.

constexpr Value exp_of(const Value &u) {
  if (u.NaN())
    return NaN();
  if (u.kind == Value::Kind::PlusInf)
    return plus_inf();
  if (u.kind == Value::Kind::MinusInf)
    return finite(0.0, Side::Above); // the rule the sigmoid turns on
  const double w = cx::exp(u.value);
  // Underflow: the values are still positive, whatever side the argument came
  // from, so Above is the side the next op should see.
  if (w == 0.0)
    return finite(0.0, Side::Above);
  return finite(w, sign_of(u.side));
}

constexpr Value log_of(const Value &u) {
  if (u.NaN())
    return NaN();
  if (u.kind == Value::Kind::PlusInf)
    return plus_inf();
  if (u.kind == Value::Kind::MinusInf)
    return NaN(); // outside the domain
  if (u.value > 0.0)
    return finite(cx::log(u.value), sign_of(u.side));
  if (u.value == 0.0 && u.side == Side::Above)
    return minus_inf();
  return NaN(); // 0 from the left, 0 exactly, or negative
}

constexpr Value sqrt_of(const Value &u) {
  if (u.NaN())
    return NaN();
  if (u.kind == Value::Kind::PlusInf)
    return plus_inf();
  if (u.kind == Value::Kind::MinusInf)
    return NaN();
  if (u.value > 0.0)
    return finite(cx::sqrt(u.value), sign_of(u.side));
  if (u.value == 0.0 && (u.side == Side::Above || u.side == Side::Exactly))
    return finite(0.0, u.side);
  return NaN();
}

constexpr Value erfc_of(const Value &u) {
  if (u.NaN())
    return NaN();
  if (u.kind == Value::Kind::PlusInf)
    return finite(0.0, Side::Above);
  if (u.kind == Value::Kind::MinusInf)
    return finite(2.0, Side::Below);
  const double w = cx::erfc(u.value);
  // erfc is decreasing, and saturates to values that are still strictly
  // inside (0, 2).
  if (w == 0.0)
    return finite(0.0, Side::Above);
  if (w == 2.0)
    return finite(2.0, Side::Below);
  return finite(w, sign_neg(sign_of(u.side)));
}

// --- the kinks -------------------------------------------------------------

constexpr Value abs_of(const Value &u) {
  if (u.NaN())
    return NaN();
  if (u.infinite())
    return plus_inf();
  if (u.value > 0.0)
    return finite(u.value, sign_of(u.side));
  if (u.value < 0.0)
    return finite(-u.value, sign_neg(sign_of(u.side)));
  // |f| at 0 is 0 from above unless f is identically 0.
  return finite(0.0, u.side == Side::Exactly ? Side::Exactly : Side::Above);
}

// When the two limits are equal the larger one near the point is whichever
// approaches from the higher side, so max takes the higher of the two sides
// (and an unknown side only spoils it when nothing outranks it).
constexpr int side_max(int p, int q) {
  if (p == q)
    return p;
  if (p == 1 || q == 1)
    return 1;
  if (!known(p) || !known(q))
    return kUnknownSign;
  return p > q ? p : q;
}

constexpr int side_min(int p, int q) {
  return sign_neg(side_max(sign_neg(p), sign_neg(q)));
}

constexpr Value max_of(const Value &u, const Value &v) {
  if (u.NaN() || v.NaN())
    return NaN();
  if (u.kind == Value::Kind::PlusInf || v.kind == Value::Kind::PlusInf)
    return plus_inf();
  if (u.kind == Value::Kind::MinusInf)
    return v;
  if (v.kind == Value::Kind::MinusInf)
    return u;
  if (u.value > v.value)
    return u;
  if (u.value < v.value)
    return v;
  return finite(u.value, side_max(sign_of(u.side), sign_of(v.side)));
}

constexpr Value min_of(const Value &u, const Value &v) {
  if (u.NaN() || v.NaN())
    return NaN();
  if (u.kind == Value::Kind::MinusInf || v.kind == Value::Kind::MinusInf)
    return minus_inf();
  if (u.kind == Value::Kind::PlusInf)
    return v;
  if (v.kind == Value::Kind::PlusInf)
    return u;
  if (u.value < v.value)
    return u;
  if (u.value > v.value)
    return v;
  return finite(u.value, side_min(sign_of(u.side), sign_of(v.side)));
}

// --- conditions ------------------------------------------------------------

enum class Truth { False, True, Unknown };

constexpr Truth truth_not(Truth t) {
  if (t == Truth::True)
    return Truth::False;
  if (t == Truth::False)
    return Truth::True;
  return Truth::Unknown;
}

constexpr Truth truth_and(Truth x, Truth y) {
  if (x == Truth::False || y == Truth::False)
    return Truth::False;
  if (x == Truth::True && y == Truth::True)
    return Truth::True;
  return Truth::Unknown;
}

constexpr Truth truth_or(Truth x, Truth y) {
  if (x == Truth::True || y == Truth::True)
    return Truth::True;
  if (x == Truth::False && y == Truth::False)
    return Truth::False;
  return Truth::Unknown;
}

// Is u < v as we approach the limit?
constexpr Truth cmp_lt(const Value &u, const Value &v) {
  if (u.NaN() || v.NaN())
    return Truth::Unknown;

  if (u.infinite() || v.infinite()) {
    if (u.kind == v.kind)
      return Truth::Unknown; // both racing to the same infinity
    if (u.kind == Value::Kind::MinusInf || v.kind == Value::Kind::PlusInf)
      return Truth::True;
    return Truth::False;
  }

  if (u.value < v.value)
    return Truth::True;
  if (u.value > v.value)
    return Truth::False;

  const int p = sign_of(u.side);
  const int q = sign_of(v.side);
  if (!known(p) || !known(q))
    return Truth::Unknown;
  if (p == q)
    // Same limit, same side: which is nearer is a rate question.
    return p == 0 ? Truth::False : Truth::Unknown;
  return p < q ? Truth::True : Truth::False;
}

constexpr Truth cmp_le(const Value &u, const Value &v) {
  if (u.NaN() || v.NaN())
    return Truth::Unknown;
  if (u.infinite() || v.infinite()) {
    if (u.kind == v.kind)
      return Truth::Unknown;
    if (u.kind == Value::Kind::MinusInf || v.kind == Value::Kind::PlusInf)
      return Truth::True;
    return Truth::False;
  }
  if (u.value < v.value)
    return Truth::True;
  if (u.value > v.value)
    return Truth::False;

  const int p = sign_of(u.side);
  const int q = sign_of(v.side);
  if (!known(p) || !known(q))
    return Truth::Unknown;
  if (p == q)
    return p == 0 ? Truth::True : Truth::Unknown;
  return p < q ? Truth::True : Truth::False;
}

constexpr Truth cmp_eq(const Value &u, const Value &v) {
  if (u.NaN() || v.NaN())
    return Truth::Unknown;
  if (u.infinite() || v.infinite())
    return Truth::Unknown;
  if (u.value != v.value)
    return Truth::False;

  const int p = sign_of(u.side);
  const int q = sign_of(v.side);
  if (!known(p) || !known(q))
    return Truth::Unknown;
  if (p == 0 && q == 0)
    return Truth::True; // both pinned to the same value
  return p != q ? Truth::False : Truth::Unknown;
}

// ---------------------------------------------------------------------------
// One walk: one direction of approach, one pass over the DAG.
// ---------------------------------------------------------------------------
struct Walk {
  Value out{};
  // Diagnostics only: the first node whose rule gave up. Ignored when the
  // output is fine, since a None on the dead side of a branch is harmless.
  int first_none_node = -1;
  OpKind first_none_op = OpKind::Input;
};

template <info Fn, std::size_t P>
consteval Walk walk(const std::array<Value, P> &inputs) {
  static constexpr auto nodes = std::define_static_array(build_nodes<Fn>());
  constexpr std::size_t N = nodes.size();

  Value vals[N];
  Truth truth[N];
  for (std::size_t i = 0; i < N; ++i) {
    vals[i] = NaN();
    truth[i] = Truth::Unknown;
  }

  Walk result{};
  auto record = [&result](std::size_t i, OpKind op) {
    if (result.first_none_node < 0) {
      result.first_none_node = static_cast<int>(i);
      result.first_none_op = op;
    }
  };

  bool saw_output = false;

  template for (constexpr auto n : nodes) {
    if constexpr (n.op == OpKind::Input) {
      if constexpr (n.self >= P)
        return Walk{NaN(), static_cast<int>(n.self), OpKind::Input};
      else
        vals[n.self] = inputs[n.self];

    } else if constexpr (n.op == OpKind::Const) {
      constexpr double v = static_cast<double>([:n.leaf:]);
      // A constant is pinned: it does not approach its value from anywhere.
      vals[n.self] = finite(v, Side::Exactly);

    } else if constexpr (n.op == OpKind::Output) {
      vals[n.self] = vals[n.a];
      result.out = vals[n.a];
      saw_output = true;

    } else if constexpr (n.op == OpKind::Add) {
      vals[n.self] = add(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Sub) {
      vals[n.self] = sub(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Mul) {
      vals[n.self] = mul(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Div) {
      vals[n.self] = div(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Neg) {
      vals[n.self] = neg(vals[n.a]);

    } else if constexpr (n.op == OpKind::Exp) {
      vals[n.self] = exp_of(vals[n.a]);
    } else if constexpr (n.op == OpKind::Log) {
      vals[n.self] = log_of(vals[n.a]);
    } else if constexpr (n.op == OpKind::Sqrt) {
      vals[n.self] = sqrt_of(vals[n.a]);
    } else if constexpr (n.op == OpKind::Erfc) {
      vals[n.self] = erfc_of(vals[n.a]);

    } else if constexpr (n.op == OpKind::Sin || n.op == OpKind::Cos) {
      // No constexpr kernel, so a finite point cannot be evaluated; at ±inf
      // there is genuinely no limit. Both come out None.
      vals[n.self] = NaN();

    } else if constexpr (n.op == OpKind::Abs) {
      vals[n.self] = abs_of(vals[n.a]);
    } else if constexpr (n.op == OpKind::Max) {
      vals[n.self] = max_of(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Min) {
      vals[n.self] = min_of(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Relu) {
      vals[n.self] = max_of(vals[n.a], finite(0.0, Side::Exactly));

    } else if constexpr (n.op == OpKind::Lt) {
      truth[n.self] = cmp_lt(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Le) {
      truth[n.self] = cmp_le(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Gt) {
      truth[n.self] = cmp_lt(vals[n.b], vals[n.a]);
    } else if constexpr (n.op == OpKind::Ge) {
      truth[n.self] = cmp_le(vals[n.b], vals[n.a]);
    } else if constexpr (n.op == OpKind::Eq) {
      truth[n.self] = cmp_eq(vals[n.a], vals[n.b]);
    } else if constexpr (n.op == OpKind::Ne) {
      truth[n.self] = truth_not(cmp_eq(vals[n.a], vals[n.b]));
    } else if constexpr (n.op == OpKind::And) {
      truth[n.self] = truth_and(truth[n.a], truth[n.b]);
    } else if constexpr (n.op == OpKind::Or) {
      truth[n.self] = truth_or(truth[n.a], truth[n.b]);
    } else if constexpr (n.op == OpKind::Not) {
      truth[n.self] = truth_not(truth[n.a]);
    } else if constexpr (n.op == OpKind::Select) {
      // Evaluate the condition as approach the limit (if possible)
      const Truth c = truth[n.cond];
      vals[n.self] = c == Truth::True    ? vals[n.a]
                     : c == Truth::False ? vals[n.b]
                                         : NaN();

    } else {
      throw "Unsupported operation";
    }

    if constexpr (!op_is_boolean(n.op)) {
      if (vals[n.self].NaN())
        record(n.self, n.op);
    }
  }

  if (!saw_output)
    return Walk{NaN(), -1, OpKind::Output};

  return result;
}

// One coordinate of the call: the operand that moves, or a pinned value.
struct Coord {
  bool is_limit = false;
  At approach{};
  double fixed = 0.0;
};

constexpr Coord to_coord(At a) { return {true, a, 0.0}; }
constexpr Coord to_coord(double v) { return {false, At{}, v}; }

template <typename T>
inline constexpr bool is_coordinate_v =
    std::is_same_v<T, At> || std::is_convertible_v<T, double>;

template <typename... Args>
inline constexpr std::size_t limit_count_v =
    ((std::is_same_v<Args, At> ? std::size_t{1} : std::size_t{0}) + ... +
     std::size_t{0});

// The moving input's element for one direction of approach.
constexpr Value approaching(const At &a, At::Dir dir) {
  switch (dir) {
  case At::Dir::FromLeft:
    return finite(a.at, Side::Below);
  case At::Dir::FromRight:
    return finite(a.at, Side::Above);
  case At::Dir::PlusInf:
    return plus_inf();
  case At::Dir::MinusInf:
    return minus_inf();
  case At::Dir::Both:
    break;
  }
  return NaN();
}

template <info Fn, std::size_t P>
consteval Walk one_sided(const std::array<Coord, P> &coords, At::Dir dir) {
  std::array<Value, P> inputs{};
  for (std::size_t i = 0; i < P; ++i)
    inputs[i] = coords[i].is_limit ? approaching(coords[i].approach, dir)
                                   : finite(coords[i].fixed, Side::Exactly);
  return walk<Fn, P>(inputs);
}

constexpr LimitResult from_walk(const Walk &w) {
  if (w.out.finite())
    return {true, w.out.value, w.out, -1, OpKind::Input};
  // An infinity is a real answer about the function, just not a convergent
  // one, so it is reported without a failing node.
  if (w.out.infinite())
    return {false, nan(), w.out, -1, OpKind::Input};
  return {false, nan(), NaN(), w.first_none_node, w.first_none_op};
}

} // namespace detail_lim

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

// The limit, the side it is approached from, and which node stopped the walk.
//
// Arguments are in parameter order, exactly one of them an ad::limits::At --
// the operand that approaches it. The others are the values their parameters
// are pinned at.
template <info Fn, typename... Args>
consteval LimitResult limit_result(Args... args) {
  static_assert((detail_lim::is_coordinate_v<Args> && ...),
                "Each argument must be an ad::limits::At or a double");
  static_assert(detail_lim::limit_count_v<Args...> == 1,
                "Exactly one argument must be an ad::limits::At: the operand "
                "that approaches it");
  constexpr std::size_t P = sizeof...(args);
  const std::array<detail_lim::Coord, P> coords{detail_lim::to_coord(args)...};

  At moving{};
  for (std::size_t i = 0; i < P; ++i)
    if (coords[i].is_limit)
      moving = coords[i].approach;

  if (moving.dir != At::Dir::Both)
    return detail_lim::from_walk(
        detail_lim::one_sided<Fn, P>(coords, moving.dir));

  // Two-sided: both one-sided limits must exist and agree.
  const detail_lim::Walk l =
      detail_lim::one_sided<Fn, P>(coords, At::Dir::FromLeft);
  const detail_lim::Walk r =
      detail_lim::one_sided<Fn, P>(coords, At::Dir::FromRight);

  if (l.out.NaN())
    return detail_lim::from_walk(l);
  if (r.out.NaN())
    return detail_lim::from_walk(r);

  if (l.out.kind != r.out.kind)
    return {false, detail_lim::nan(), detail_lim::NaN(), -1, OpKind::Select};

  if (l.out.infinite())
    return {false, detail_lim::nan(), l.out, -1, OpKind::Input};

  if (l.out.value != r.out.value)
    // A jump: both sides have a limit, they just are not the same one.
    return {false, detail_lim::nan(), detail_lim::NaN(), -1, OpKind::Select};

  // Approached from opposite sides, so the merged side is only meaningful when
  // the two walks agree on it.
  const Side side = l.out.side == r.out.side ? l.out.side : Side::Unknown;
  const Value merged{Value::Kind::Finite, l.out.value, side};
  return {true, merged.value, merged, -1, OpKind::Input};
}

// Returns true iff the function reflected by Fn has a finite limit as the
// operand marked with an ad::limits::At approaches it.
template <info Fn, typename... Args>
consteval bool is_convergent_at(Args... args) {
  return limit_result<Fn, Args...>(args...).converges;
}

// The limit itself. Ill-formed unless the limit is provable, so a value is
// never silently invented -- mirroring ad::inverse and ad::limit_of.
template <info Fn, typename... Args> consteval double limit_of(Args... args) {
  const LimitResult r = limit_result<Fn, Args...>(args...);
  if (!r.converges)
    throw "ad::limits::limit_of requires a provable finite limit; "
          "is_convergent_at is false for this function and limit point";
  return r.value;
}

} // namespace ad::limits

#endif // LIMIT_ALGEBRA_HPP
