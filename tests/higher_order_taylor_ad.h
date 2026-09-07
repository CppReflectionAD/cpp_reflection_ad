#ifndef REFLECT_DEMO_HIGHER_ORDER_TAYLOR_AD_H
#define REFLECT_DEMO_HIGHER_ORDER_TAYLOR_AD_H

#include <meta>

#include "autograd.h"

namespace ad {

using std::meta::info;

// ---------------------------------------------------------------------------
// Higher-order derivatives by truncated Taylor propagation.
// ---------------------------------------------------------------------------
namespace detail_taylor_ad {

// Raw append: always pushes a new node and returns its slot.
consteval std::size_t emit_raw(std::vector<Node> &out, OpKind op, std::size_t a,
                               std::size_t b, info leaf = ^^int,
                               std::size_t cond = 0,
                               std::size_t guard = UNGUARDED) {
  std::size_t s = out.size();
  out.push_back(Node{op, s, a, b, leaf, cond, guard});
  return s;
}

// If we have an existing node in the DAG with the same operation and operands,
// we can reuse it instead of bloating the DAG with duplicate nodes. Const nodes
// carry their value in `leaf` rather than in operands, so they only match when
// the leaf matches too. `cond` and `guard` are also keyed on since a value
// computed under one guard must not be reused under another.
consteval std::size_t emit_node(std::vector<Node> &out, OpKind op,
                                std::size_t a, std::size_t b, info leaf = ^^int,
                                std::size_t cond = 0,
                                std::size_t guard = UNGUARDED) {
  for (const Node &n : out) {
    if (n.op == op && n.a == a && n.b == b && n.cond == cond &&
        n.guard == guard && (op != OpKind::Const || n.leaf == leaf)) {
      return n.self;
    }
  }
  return emit_raw(out, op, a, b, leaf, cond, guard);
}

// This struct represents the shape of a truncated taylor polynomial, e.g.
// f(x, y) = f(a, b) + fx(a, b)(x-a) + fy(a, b)(y-b) + 1/2 * (fxx(a,b)(x-a)^2 +
// 2fxy(a,b)(x-a)(y-b) + fyy(a,b)(y-b)^2) Let's say f(x, y) is a combination of
// unary functions, and we want to calculate fxy(a, b) (1st derivative wrt x and
// y) We can look at the coefficient of (x-a)(y-b) in the taylor expansion of
// f(x, y) to work out fxy. e.g. f(x,y) = sin(x)cos(y) = sin(a)cos(b) +
// cos(a)cos(b)(x-a) - sin(a)sin(b)(y-b) + 0.5 (-sin(a)cos(b)(x-a)^2 -
// 2cos(a)sin(b)(x-a)^2 -sin(a)cos(b)(y-b)^2) we see the coefficient is
// -cos(a)sin(b) We can work this out in a simpler way, by considering f(x, y) =
// g(x)h(y) Where g(x) = sin(x) = sin(a) + cos(a)(x-a)
//       h(y) = cos(y) = cos(b) - sin(b)(y-b)
// We can multiply these together, and look at the coefficient of (x-a)(y-b) to
// get the same result. We use this trick to work out higher order complex
// derivatives, by composing simpler taylor polynomials.
struct PolynomialShape {
  // slot in the DAG of each argument we want to differentiate with respect to
  // (e.g. for polynomial above, slot of x and y)
  std::vector<std::size_t> wrtSlots;
  // how many terms in each polynomial are required for each slot. (e.g. above,
  // we only needed 2 terms in the expansion of g(x) and h(y)) in general, we
  // need the number of times we want to differentiate w.r.t + 1.
  std::vector<std::size_t> wrtTerms;
  // The index of the term in the polynomial that corresponds to the first order
  // derivative for each w.r.t. e.g. for the example above firstDerivative =
  // [1,2]
  std::vector<std::size_t> firstDerivative;
  // The order of each partial derivative for each term in the taylor polynomial
  // For instance in the example above, orders = {{0,0}, {1, 0}, {0, 1}, {1,
  // 1}}.
  std::vector<std::vector<std::size_t>> orders;
  // how many terms are in the overall taylor polynomial (each wrt polynomical
  // multiplied together)
  std::size_t totalTerms = 1;
  // the order of the overall taylor polynomial (e.g. x*x*y = 3rd order)
  std::size_t polynomialOrder = 0;

  // For a polynomial with shape `sh`, is the algebraic term at index `a` a
  // higher order than that at index `b`. Here a higher oder means that `b`
  // divides `a`, e.g. xy^2 is a divisor of x^2y^3, but x^3y^2 is not.
  consteval bool isPolynomialDivisor(std::size_t a, std::size_t b) const {
    for (std::size_t i = 0; i < orders[a].size(); ++i) {
      if (orders[b][i] > orders[a][i]) {
        return false;
      }
    }
    return true;
  }

  // Create a `PolynomialShape` for a list of variables we want to differentiate
  // `wrt`.
  static consteval PolynomialShape
  make_shape(const std::vector<std::size_t> &wrts) {
    PolynomialShape sh;
    for (std::size_t w : wrts) {
      bool seen = false;
      for (std::size_t i = 0; i < sh.wrtSlots.size(); ++i) {
        if (sh.wrtSlots[i] == w) {
          ++sh.wrtTerms[i];
          seen = true;
          break;
        }
      }
      if (!seen) {
        sh.wrtSlots.push_back(w);
        sh.wrtTerms.push_back(2);
      }
    }
    for (std::size_t r : sh.wrtTerms) {
      sh.firstDerivative.push_back(sh.totalTerms);
      sh.totalTerms *= r;
    }
    sh.polynomialOrder = wrts.size();
    sh.orders.resize(sh.totalTerms);
    for (std::size_t termIndex = 0; termIndex < sh.totalTerms; ++termIndex) {
      std::vector<std::size_t> degrees(sh.wrtSlots.size(), 0);
      std::size_t t = termIndex;
      // Arrange the polynomial degrees such that the term at index i can be
      // created by multiplying terms at index j and (i-j) as long as the term
      // at index j divides that at index i. e.g. at index j we might have x^2 *
      // y^2. At index i we have x^4 * y^5. Then at index i-j, we have x^2 *
      // y^3. This is achieved by choosing the degrees via the `%` operator
      // below.
      for (std::size_t i = 0; i < sh.wrtSlots.size(); ++i) {
        degrees[i] = t % sh.wrtTerms[i];
        t /= sh.wrtTerms[i];
      }
      sh.orders[termIndex] = degrees;
    }
    return sh;
  }
};

struct DAGBuilder {
public:
  // Sentinel value to indicate that a derivative is zero and does not need
  // adding to the DAG.
  static constexpr std::size_t kNoSlot = static_cast<std::size_t>(-1);

private:
  // DAG slot of literal 1.0, if it's in the DAG.
  std::size_t one = kNoSlot;

  // A compile-time map from a constant scalar value to the slot of its (unique)
  // Const node.
  using ConstPool = std::vector<std::pair<double, std::size_t>>;
  ConstPool pool;

public:
  std::vector<Node> out;

  // The guard every node emitted from here on is predicated on.
  std::size_t curGuard = UNGUARDED;

  // Return the slot of the Const node holding `value`. Cache it and ensure that
  // there are no duplicated constants in the output nodes. Constants do not
  // need guarding, so ignore the `curGuard`.
  consteval std::size_t ensure_const_node(double value) {
    for (const auto &[v, slot] : pool) {
      if (v == value) {
        return slot;
      }
    }
    std::size_t slot =
        emit_raw(out, OpKind::Const, 0, 0, m::reflect_constant(value));
    pool.push_back({value, slot});
    return slot;
  }

  // Add a node for `op` applied to the existing nodes at `slot1` (and, for a
  // binary op, `slot2`), under the guard currently in force.
  consteval std::size_t emit_op_node(OpKind op, std::size_t slot1,
                                     std::size_t slot2 = 0) {
    return emit_node(out, op, slot1, slot2, ^^int, 0, curGuard);
  }

  // Add a negation to the `out` vector, negating the existing node at position
  // `slot`.
  consteval std::size_t emit_negation_node(std::size_t slot) {
    return slot == DAGBuilder::kNoSlot ? DAGBuilder::kNoSlot
                                       : emit_op_node(OpKind::Neg, slot, 0);
  }

  // Add an addition to the `out` vector, adding the existing nodes at positions
  // `slot1` and `slot2`.
  consteval std::size_t emit_addition_node(std::size_t slot1,
                                           std::size_t slot2) {
    if (slot1 == DAGBuilder::kNoSlot) {
      return slot2;
    }
    if (slot2 == DAGBuilder::kNoSlot) {
      return slot1;
    }
    return emit_op_node(OpKind::Add, slot1, slot2);
  }

  // Add an subtraction to the `out` vector, subtracting the existing nodes at
  // positions `slot1` and `slot2`.
  consteval std::size_t emit_subtraction_node(std::size_t slot1,
                                              std::size_t slot2) {
    if (slot2 == DAGBuilder::kNoSlot) {
      return slot1;
    }
    if (slot1 == DAGBuilder::kNoSlot) {
      return emit_negation_node(slot2);
    }
    return emit_op_node(OpKind::Sub, slot1, slot2);
  }

  // Add a multiplication to the `out` vector, multiplying the existing nodes at
  // positions `slot1` and `slot2`.
  consteval std::size_t emit_multiplication_node(std::size_t slot1,
                                                 std::size_t slot2) {
    if (slot1 == DAGBuilder::kNoSlot || slot2 == DAGBuilder::kNoSlot) {
      return DAGBuilder::kNoSlot;
    }
    if (slot1 == one) {
      return slot2;
    }
    if (slot2 == one) {
      return slot1;
    }
    return emit_op_node(OpKind::Mul, slot1, slot2);
  }

  // Add a division to the `out` vector, dividing the existing nodes at
  // positions `slot1` and `slot2`.
  consteval std::size_t emit_division_node(std::size_t slot1,
                                           std::size_t slot2) {
    if (slot2 == DAGBuilder::kNoSlot) {
      throw "reflection AD: division by a structurally zero value";
    }
    if (slot1 == DAGBuilder::kNoSlot) {
      return DAGBuilder::kNoSlot;
    }
    if (slot2 == one) {
      return slot1;
    }
    return emit_op_node(OpKind::Div, slot1, slot2);
  }

  // Add a multiplcation to the `out` vector, scaling the node at position
  // `slot` by the `scaleFactor`.
  consteval std::size_t emit_scaling_node(std::size_t slot,
                                          double scaleFactor) {
    if (slot == DAGBuilder::kNoSlot || scaleFactor == 0.0) {
      return DAGBuilder::kNoSlot;
    }
    if (scaleFactor == 1.0) {
      return slot;
    }
    if (scaleFactor == -1.0) {
      return emit_negation_node(slot);
    }
    return emit_op_node(OpKind::Mul, ensure_const_node(scaleFactor), slot);
  }

  // Add a `cond ? slot1 : slot2` to the `out` vector.
  consteval std::size_t emit_select_node(std::size_t cond, std::size_t slot1,
                                         std::size_t slot2) {
    if (slot1 == DAGBuilder::kNoSlot && slot2 == DAGBuilder::kNoSlot) {
      return DAGBuilder::kNoSlot;
    }
    std::size_t whenTrue =
        slot1 == DAGBuilder::kNoSlot ? ensure_const_node(0.0) : slot1;
    std::size_t whenFalse =
        slot2 == DAGBuilder::kNoSlot ? ensure_const_node(0.0) : slot2;
    if (whenTrue == whenFalse) {
      // Both branches are the same, so no need to add a new node.
      return whenTrue;
    }
    return emit_node(out, OpKind::Select, whenTrue, whenFalse, ^^int, cond,
                     curGuard);
  }

  // Return the slot in the DAG corresponding to the literal `1.0`.
  consteval std::size_t literal_one() {
    one = ensure_const_node(1.0);
    return one;
  }

  // For the DAG node at position `slot`, generate DAG nodes corresponding to
  // powers of the DAG node at `slot` up to and including the `order`th power.
  // Return a vector containing the position of these nodes in the DAG.
  consteval std::vector<std::size_t> pow_slots(std::size_t slot,
                                               std::size_t order) {
    std::vector<std::size_t> powers;
    powers.push_back(literal_one());
    for (std::size_t m = 1; m <= order; ++m) {
      powers.push_back(emit_multiplication_node(powers[m - 1], slot));
    }
    return powers;
  }
};

// One slot per multi-index, in rank order; DAGBuilder::kNoSlot where the
// coefficient is statically zero.
using Polynomial = std::vector<std::size_t>;

struct PolynomialBuilder {
  // Create a polynomial of the specified shape, with every coefficient zero.
  static consteval Polynomial
  createZeroPolynomial(const PolynomialShape &shape) {
    return Polynomial(shape.totalTerms, DAGBuilder::kNoSlot);
  }

  // Multiply polynomials `A` and `B`, both of the shape `sh` and return the
  // result. All coefficients of the result are added to the `builder`.
  static consteval Polynomial multiplyPolynomials(DAGBuilder &builder,
                                                  const PolynomialShape &sh,
                                                  const Polynomial &A,
                                                  const Polynomial &B) {
    Polynomial result = PolynomialBuilder::createZeroPolynomial(sh);
    // Calculate each term in the result polynomial one by one.
    for (std::size_t currentTerm = 0; currentTerm < sh.totalTerms;
         ++currentTerm) {
      // calculate the current coefficient by looking through all terms in both
      // polynomials that are factors. e.g. (1 + 3x + 4x^2) * (2 + 2x + 5x^2)
      // the coefficient of x^2 in the result comes from (1*5 + 3*2 + 4 * 2)
      std::size_t coefficient = DAGBuilder::kNoSlot;
      for (std::size_t factorTerm = 0; factorTerm <= currentTerm;
           ++factorTerm) {
        if (A[factorTerm] == DAGBuilder::kNoSlot ||
            B[currentTerm - factorTerm] == DAGBuilder::kNoSlot ||
            !sh.isPolynomialDivisor(currentTerm, factorTerm)) {
          continue;
        }
        // by construction, (polynomial power at index b) * (polynomial power at
        // index a - b) = (polynomial power at index a)
        coefficient = builder.emit_addition_node(
            coefficient, builder.emit_multiplication_node(
                             A[factorTerm], B[currentTerm - factorTerm]));
      }
      result[currentTerm] = coefficient;
    }
    return result;
  }

  // Divide polynomial `A` by `B`, both of the shape `sh` and return the result.
  // All coefficients of the result are added to the `builder`.
  static consteval Polynomial dividePolynomials(DAGBuilder &builder,
                                                const PolynomialShape &sh,
                                                const Polynomial &A,
                                                const Polynomial &B) {
    if (B[0] == DAGBuilder::kNoSlot) {
      throw "reflection AD: division by a structurally zero value";
    }
    Polynomial result = PolynomialBuilder::createZeroPolynomial(sh);
    // Start with the 0th order term of `A`. Only the 0th order term of `B` can
    // divide it. This fixes the 0th order term in the result. We can then
    // consider the 1st order term of `A`. Only the 0th and 1st order terms of
    // `B` can divide it, where the first order term of `B` is multiplied by the
    // 0th order term of `result`, and the 0th order term of `B` is multiplied
    // by the 1st order term of result. We use this to work out the coefficient
    // of the first order term of result. We can use a similar pattern to work
    // out all coefficients.
    for (std::size_t termIndex = 0; termIndex < sh.totalTerms; ++termIndex) {
      std::size_t coefficient = A[termIndex];
      for (std::size_t factorTerm = 1; factorTerm <= termIndex; ++factorTerm) {
        if (B[factorTerm] == DAGBuilder::kNoSlot ||
            result[termIndex - factorTerm] == DAGBuilder::kNoSlot ||
            !sh.isPolynomialDivisor(termIndex, factorTerm)) {
          continue;
        }
        coefficient = builder.emit_subtraction_node(
            coefficient, builder.emit_multiplication_node(
                             B[factorTerm], result[termIndex - factorTerm]));
      }
      result[termIndex] = builder.emit_division_node(coefficient, B[0]);
    }
    return result;
  }
};

// Return the coefficients of the taylor expansion for the unary function `op`
// around `a0`, up to the `order` specified.
consteval std::vector<std::size_t>
calculateTaylorCoefficients(DAGBuilder &builder, OpKind op, std::size_t a0,
                            std::size_t order) {
  // Coefficients of the taylor polynomial.
  std::vector<std::size_t> coefficients(order + 1, DAGBuilder::kNoSlot);

  // Create a vector, where index i = 1/i!
  std::vector<double> inv_factorial(order + 1, 1.0);
  for (std::size_t m = 1; m <= order; ++m) {
    inv_factorial[m] = inv_factorial[m - 1] / static_cast<double>(m);
  }

  switch (op) {
  case OpKind::Exp: {
    // exp(x) = exp(a0) + exp(a0)/1! * (x-a0) + exp(a0)/2! * (x-a0)^2 ...
    std::size_t e = builder.emit_op_node(OpKind::Exp, a0);
    for (std::size_t m = 0; m <= order; ++m)
      coefficients[m] = builder.emit_scaling_node(e, inv_factorial[m]);
    break;
  }
  case OpKind::Sin:
  case OpKind::Cos: {
    // sin(x) = sin(a0) + cos(a0)/1! * (x-a0) - sin(a0)/2! * (x-a0)^2 ...
    // cos(x) = cos(a0) - sin(a0)/1! * (x-a0) - cos(a0)/2! * (x-a0)^2 ...
    std::size_t s = builder.emit_op_node(OpKind::Sin, a0);
    std::size_t k = builder.emit_op_node(OpKind::Cos, a0);
    for (std::size_t m = 0; m <= order; ++m) {
      std::size_t phase = ((op == OpKind::Sin ? m : m + 1) % 4);
      std::size_t base = (phase % 2 == 0) ? s : k;
      double sign = (phase < 2) ? 1.0 : -1.0;
      coefficients[m] =
          builder.emit_scaling_node(base, sign * inv_factorial[m]);
    }
    break;
  }
  case OpKind::Log: {
    // log(x) = log(a0) + 1/(1! * a0) * (x-a0) - 1/(2! * a0^2) * (x-a0)^2 ... +
    // (-1)^(n+1)/(n * a0^n) * (x-a0)^n
    coefficients[0] = builder.emit_op_node(OpKind::Log, a0);
    std::vector<std::size_t> p = builder.pow_slots(a0, order);
    for (std::size_t m = 1; m <= order; ++m) {
      double k = ((m % 2) ? 1.0 : -1.0) / static_cast<double>(m);
      coefficients[m] =
          builder.emit_division_node(builder.ensure_const_node(k), p[m]);
    }
    break;
  }
  case OpKind::Sqrt: {
    // sqrt(x) = sqrt(a0) + 1/1!*sqrt(a0)/2*a0 (x-a0) + 1/2!*sqrt(a0)/4*a0^2
    // (x-a0)^2 +... + 1/n!(-1)^n-1(2n-3)!/2^n * sqrt(a0)/a0^n * (x-a0)^n
    std::size_t s = builder.emit_op_node(OpKind::Sqrt, a0);
    coefficients[0] = s;
    // Need to divide by a0^n for the nth term, so calculate powers.
    std::vector<std::size_t> powers = builder.pow_slots(a0, order);
    double coefficient = 1.0;
    for (std::size_t m = 1; m <= order; ++m) {
      coefficient *=
          -0.5 * static_cast<double>(2.0 * m - 3.0) / static_cast<double>(m);
      coefficients[m] = builder.emit_scaling_node(
          builder.emit_division_node(s, powers[m]), coefficient);
    }
    break;
  }
  case OpKind::Erfc: {
    // mth derivative of erfc:
    // erfc^(m)(a) = -2/sqrt(pi) * (-1)^(m-1) * H_{m-1}(a) * exp(-a*a)
    // where H_{m-1} is the m-1th physicist's Hermite polynomial
    // and H_{n+1}(x) = 2x * H_{n}(x) - 2*n*H_{n-1}(x)
    // and H_0(x) = 1, H_1(x) = 2x
    coefficients[0] = builder.emit_op_node(OpKind::Erfc, a0);
    if (order >= 1) {
      std::size_t square = builder.emit_op_node(OpKind::Mul, a0, a0);
      std::size_t e = builder.emit_op_node(
          OpKind::Exp, builder.emit_op_node(OpKind::Neg, square));
      std::vector<std::size_t> hermitePolynomials(order, DAGBuilder::kNoSlot);
      hermitePolynomials[0] = builder.literal_one(); // H_0 = 1
      if (order >= 2)
        hermitePolynomials[1] = builder.emit_scaling_node(a0, 2.0); // H_1 = 2a
      for (std::size_t n = 2; n < order; ++n) {
        // Calculate each Hermite polynomial.
        std::size_t two_times_a = builder.emit_scaling_node(a0, 2.0);
        std::size_t part1 = builder.emit_multiplication_node(
            two_times_a, hermitePolynomials[n - 1]);
        std::size_t part2 = builder.emit_scaling_node(
            hermitePolynomials[n - 2], 2.0 * static_cast<double>(n - 1));
        hermitePolynomials[n] = builder.emit_subtraction_node(part1, part2);
      }

      for (std::size_t m = 1; m <= order; ++m) {
        double k =
            -two_over_root_pi * ((m % 2) ? 1.0 : -1.0) * inv_factorial[m];
        coefficients[m] = builder.emit_scaling_node(
            builder.emit_multiplication_node(hermitePolynomials[m - 1], e), k);
      }
    }
    break;
  }
  default:
    throw "reflection AD: unsupported operation";
  }
  return coefficients;
}

// Use the chain rule to calculate the coefficients of a Taylor polynomial for
// the Operation `op` around A[0], where `A` is the taylor polynomial of the
// input of `op`. We can substitute in the taylor polynomial for `A` into the
// taylor expansion for `op`. E.g. exp(sin(x))
//  exp(x) = exp(a0) + exp(a0)/1! * (x-a0) + exp(a0)/2! * (x-a0)^2 ...   (1)
//  sin(y) = sin(b0) + cos(b0)/1! * (y-b0) - sin(b0)/2! * (y-b0)^2 ...   (2)
// Putting x = sin(y), a0 = sin(b0) into (1)
//  exp(sin(y)) = exp(sin(b0)) + exp(sin(b0))/1! * (cos(b0)/1! * (y-b0) -
//  sin(b0)/2! * (y-b0)^2) + exp(sin(b0))/2! + (cos(b0)/1! * (y-b0) - sin(b0)/2!
//  * (y-b0)^2) ^ 2 ...
//              = exp(sin(b0)) + exp(sin(b0))/1! * (cos(b0)/1! * (y-b0)) to
//              first order.
consteval Polynomial apply_unary_chain_rule(DAGBuilder &builder,
                                            const PolynomialShape &sh,
                                            OpKind op, const Polynomial &A) {
  std::vector<std::size_t> coefficients =
      calculateTaylorCoefficients(builder, op, A[0], sh.polynomialOrder);
  Polynomial result = PolynomialBuilder::createZeroPolynomial(sh);
  result[0] = coefficients[0];

  // Let P be the taylor polynomial of the input to `op`, `A`, expanded around
  // A[0]
  Polynomial P = A;
  // Set P[0] to 0, meaning P = A - A[0].
  P[0] = DAGBuilder::kNoSlot;

  // We can calculate the taylor polynomial R: sum( coefficients[i] * (A - A[0])
  // ^ i-1 ) We can substitute in P to get the taylor polynomial for R in terms
  // of the inputs to A.
  Polynomial pw = PolynomialBuilder::createZeroPolynomial(sh);
  pw[0] = builder.literal_one(); // P^0
  // For each polynomial order we care about, multiple `pw` by `P` and update
  // the coefficients. We need
  //  to do this for each power, since for instance if P = x + x^2, and we care
  //  about
  // 4th order terms, we'll get contributions from P^2 and P^4.
  for (std::size_t m = 1; m <= sh.polynomialOrder; ++m) {
    pw = PolynomialBuilder::multiplyPolynomials(builder, sh, pw, P);
    bool any = false;
    for (std::size_t r = 0; r < sh.totalTerms; ++r) {
      if (pw[r] != DAGBuilder::kNoSlot) {
        any = true;
        break;
      }
    }
    if (!any) {
      break;
    }
    for (std::size_t r = 0; r < sh.totalTerms; ++r) {
      result[r] = builder.emit_addition_node(
          result[r], builder.emit_multiplication_node(coefficients[m], pw[r]));
    }
  }
  return result;
}

// Build the DAG computing d^|Wrts| f / prod(d x_w). For each node in `src`,
// compute the taylor expansion of the required order. Combine these taylor
// polynomials according to the DAG structure and extract the derivatives from
// the result.
consteval std::vector<Node>
build_taylor_nodes(const std::vector<Node> &src,
                   const std::vector<std::size_t> &wrts) {
  PolynomialShape sh = PolynomialShape::make_shape(wrts);
  DAGBuilder builder;

  // Inputs come first and keep their slots: the evaluator reads `in[n.self]`.
  // Emitting them before any constant is what preserves that.
  for (const Node &n : src) {
    if (n.op == OpKind::Input) {
      std::size_t s = emit_raw(builder.out, OpKind::Input, 0, 0);
      if (s != n.self)
        throw "reflection AD: input nodes are not at the head of the DAG";
    }
  }

  // Create a taylor polynomial for each node in the src.
  std::vector<Polynomial> polynomials(src.size());
  for (const Node &n : src) {
    Polynomial current = PolynomialBuilder::createZeroPolynomial(sh);
    // The coefficient arithmetic for this node belongs to the same branch the
    // node itself does. `src` guards are slots in `src`; the corresponding
    // value in the DAG being built is the guard's 0th coefficient, and guards
    // precede what they guard, so it is already there.
    builder.curGuard =
        n.guard == UNGUARDED ? UNGUARDED : polynomials[n.guard][0];
    switch (n.op) {
    case OpKind::Input:
      // Construct the taylor polynomial for the input node. The first term of
      // the polynomial is the value itself. The first derivative is 1, if we
      // differentiate this argument and every other term is 0.
      current[0] = n.self;
      for (std::size_t i = 0; i < sh.wrtSlots.size(); ++i) {
        if (sh.wrtSlots[i] == n.self) {
          current[sh.firstDerivative[i]] = builder.literal_one();
        }
      }
      break;
    case OpKind::Const:
      // We don't use the DAG builder const pool here, as we wish to preserve
      // the constant from the original calculation.
      current[0] = emit_raw(builder.out, OpKind::Const, 0, 0, n.leaf);
      break;
    case OpKind::Output:
      // Add a new polynomial to indicate the output.
      current = polynomials[n.a];
      break;
    case OpKind::Add:
      // Add the coefficients of the two polynomials together.
      for (std::size_t r = 0; r < sh.totalTerms; ++r) {
        current[r] = builder.emit_addition_node(polynomials[n.a][r],
                                                polynomials[n.b][r]);
      }
      break;
    case OpKind::Sub:
      // Subtract the coefficients of the second polynomial from the first.
      for (std::size_t r = 0; r < sh.totalTerms; ++r) {
        current[r] = builder.emit_subtraction_node(polynomials[n.a][r],
                                                   polynomials[n.b][r]);
      }
      break;
    case OpKind::Neg:
      // Negate every coefficient of the polynomial.
      for (std::size_t r = 0; r < sh.totalTerms; ++r) {
        current[r] = builder.emit_negation_node(polynomials[n.a][r]);
      }
      break;
    case OpKind::Mul:
      current = PolynomialBuilder::multiplyPolynomials(
          builder, sh, polynomials[n.a], polynomials[n.b]);
      break;
    case OpKind::Div:
      current = PolynomialBuilder::dividePolynomials(
          builder, sh, polynomials[n.a], polynomials[n.b]);
      break;
    case OpKind::Lt:
    case OpKind::Le:
    case OpKind::Gt:
    case OpKind::Ge:
    case OpKind::Eq:
    case OpKind::Ne:
    case OpKind::And:
    case OpKind::Or:
    case OpKind::Not:
      // Propagate the condition, acting on the 0th order term. Higher orders
      // are zero.
      current[0] = builder.emit_op_node(
          n.op, polynomials[n.a][0], op_has_b(n.op) ? polynomials[n.b][0] : 0);
      break;
    case OpKind::Select:
      // The current polynomial is set to the selected polynomial.
      for (std::size_t r = 0; r < sh.totalTerms; ++r)
        current[r] = builder.emit_select_node(
            polynomials[n.cond][0], polynomials[n.a][r], polynomials[n.b][r]);
      break;
    case OpKind::Abs: {
      // Take the absolute value of the input polynomial.
      const Polynomial &A = polynomials[n.a];
      std::size_t isNeg = builder.emit_op_node(OpKind::Lt, A[0],
                                               builder.ensure_const_node(0.0));
      current[0] = builder.emit_op_node(OpKind::Abs, A[0]);
      for (std::size_t r = 1; r < sh.totalTerms; ++r)
        current[r] = builder.emit_select_node(
            isNeg, builder.emit_negation_node(A[r]), A[r]);
      break;
    }
    case OpKind::Max:
    case OpKind::Min: {
      // Take whichever polynomial has a higher 0th order term.
      const Polynomial &A = polynomials[n.a];
      const Polynomial &B = polynomials[n.b];
      std::size_t cmp = (n.op == OpKind::Max)
                            ? builder.emit_op_node(OpKind::Lt, A[0], B[0])
                            : builder.emit_op_node(OpKind::Lt, B[0], A[0]);
      current[0] = builder.emit_op_node(n.op, A[0], B[0]);
      for (std::size_t r = 1; r < sh.totalTerms; ++r)
        current[r] = builder.emit_select_node(cmp, B[r], A[r]);
      break;
    }
    default: // unary primitives (throws if not)
      current = apply_unary_chain_rule(builder, sh, n.op, polynomials[n.a]);
      break;
    }
    polynomials[n.self] = current;
  }

  // The root is outside every branch, so the scaling below is unconditional.
  builder.curGuard = UNGUARDED;

  // Output polynomial is the last one.
  const Polynomial &outputPolynomial = polynomials[src.size() - 1];

  // Requested derivative is the highest power in the polynomial by
  // construction.
  std::size_t top = outputPolynomial[sh.totalTerms - 1];
  // Coefficient of the highest power in the polynomial is of the form
  // 1/k!(d^k/dx) * 1/j!(d^j/dy)... So need to multiply out the factorials to
  // retrieve the derivative.
  double factorials = 1.0;
  for (std::size_t r : sh.wrtTerms) {
    for (std::size_t k = 2; k < r; ++k) {
      factorials *= static_cast<double>(k);
    }
  }

  // Add a node to the DAG to represent the output derivative.
  std::size_t res = builder.emit_scaling_node(top, factorials);
  if (res == DAGBuilder::kNoSlot) // derivative is statically zero
    res = builder.ensure_const_node(0.0);
  emit_raw(builder.out, OpKind::Output, res, 0);
  return builder.out;
}

// Mark only nodes reachable from the Output as needed, so the evaluator
// skips nodes that are not needed.
consteval void prune_reachable(std::vector<Node> &ns) {
  const std::size_t N = ns.size();
  std::vector<char> need(N, 0);
  need[N - 1] = 1;
  for (std::size_t k = N; k-- > 0;) {
    if (!need[k])
      continue;
    const Node &n = ns[k];
    if (op_has_a(n.op))
      need[n.a] = 1;
    if (op_has_b(n.op))
      need[n.b] = 1;
    if (op_has_cond(n.op))
      need[n.cond] = 1;
    if (n.guard != UNGUARDED)
      need[n.guard] = 1;
  }
  for (Node &n : ns)
    n.nself = need[n.self];
}

// Create nodes for a DAG corresponding to the partial derivative of `Fn`,
// differentiated with respect to each argument in `Wrts...`.
template <info Fn, std::size_t... Wrts>
consteval std::vector<Node> build_partial_taylor_nodes() {
  std::vector<Node> ns = detail_taylor_ad::build_taylor_nodes(
      build_nodes<Fn>(), std::vector<std::size_t>{Wrts...});
  detail_taylor_ad::prune_reachable(ns);
  return ns;
}

} // namespace detail_taylor_ad

// Partial derivative of `Fn` with respect to each index in `Wrts...`, evaluated
// at `Arg...`.
template <info Fn, std::size_t... Wrts, typename... Args>
constexpr double taylor_mode_ad(Args... args) {
  static constexpr auto nodes = std::define_static_array(
      detail_taylor_ad::build_partial_taylor_nodes<Fn, Wrts...>());
  constexpr std::size_t N = nodes.size();
  const double in[] = {static_cast<double>(args)...};
  double val[N] = {};
  template for (constexpr auto n : nodes) {
    if constexpr (n.nself) {
      // A guarded node belongs to a branch; skip it unless that branch is
      // taken.
      if constexpr (n.guard != UNGUARDED) {
        if (!(val[n.guard] != 0.0))
          continue;
      }
      if constexpr (n.op == OpKind::Input)
        val[n.self] = in[n.self];
      else if constexpr (n.op == OpKind::Const)
        val[n.self] = static_cast<double>([:n.leaf:]);
      else if constexpr (n.op == OpKind::Output)
        val[n.self] = val[n.a];
      else if constexpr (n.op == OpKind::Add)
        val[n.self] = val[n.a] + val[n.b];
      else if constexpr (n.op == OpKind::Sub)
        val[n.self] = val[n.a] - val[n.b];
      else if constexpr (n.op == OpKind::Mul)
        val[n.self] = val[n.a] * val[n.b];
      else if constexpr (n.op == OpKind::Div)
        val[n.self] = val[n.a] / val[n.b];
      else if constexpr (n.op == OpKind::Neg)
        val[n.self] = -val[n.a];
      else if constexpr (n.op == OpKind::Sin)
        val[n.self] = std::sin(val[n.a]);
      else if constexpr (n.op == OpKind::Cos)
        val[n.self] = std::cos(val[n.a]);
      else if constexpr (n.op == OpKind::Exp)
        val[n.self] = std::exp(val[n.a]);
      else if constexpr (n.op == OpKind::Log)
        val[n.self] = std::log(val[n.a]);
      else if constexpr (n.op == OpKind::Sqrt)
        val[n.self] = std::sqrt(val[n.a]);
      else if constexpr (n.op == OpKind::Erfc)
        val[n.self] = std::erfc(val[n.a]);
      else if constexpr (n.op == OpKind::Lt)
        val[n.self] = (val[n.a] < val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Le)
        val[n.self] = (val[n.a] <= val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Gt)
        val[n.self] = (val[n.a] > val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Ge)
        val[n.self] = (val[n.a] >= val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Eq)
        val[n.self] = (val[n.a] == val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Ne)
        val[n.self] = (val[n.a] != val[n.b]) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Not)
        val[n.self] = (val[n.a] != 0.0) ? 0.0 : 1.0;
      // Reads val[b] only when val[a] leaves it undecided -- exactly when the
      // right operand was lowered as reachable, so its slot is written.
      else if constexpr (n.op == OpKind::And)
        val[n.self] = (val[n.a] != 0.0 && val[n.b] != 0.0) ? 1.0 : 0.0;
      else if constexpr (n.op == OpKind::Or)
        val[n.self] = (val[n.a] != 0.0 || val[n.b] != 0.0) ? 1.0 : 0.0;
      // Select likewise reads only the branch it takes.
      else if constexpr (n.op == OpKind::Select)
        val[n.self] = (val[n.cond] != 0.0) ? val[n.a] : val[n.b];
      else if constexpr (n.op == OpKind::Abs)
        val[n.self] = (val[n.a] < 0.0) ? -val[n.a] : val[n.a];
      else if constexpr (n.op == OpKind::Max)
        val[n.self] = (val[n.a] < val[n.b]) ? val[n.b] : val[n.a];
      else if constexpr (n.op == OpKind::Min)
        val[n.self] = (val[n.b] < val[n.a]) ? val[n.b] : val[n.a];
    }
  }
  return val[N - 1];
}

} // namespace ad

#endif // REFLECT_DEMO_HIGHER_ORDER_TAYLOR_AD_H
