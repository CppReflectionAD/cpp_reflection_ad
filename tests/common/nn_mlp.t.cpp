// The include/nn library: reverse-mode AD of tensor-level code by body
// reflection. Gradients are checked against central finite differences, the
// shape of the reflected DAG is checked at compile time, and a small MLP is
// trained on XOR. A test, not a benchmark: it runs in milliseconds.

#include <nn/nn.hpp>
#include <test_simple_include.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>

// Dynamic extents: heap storage, so a non-trivial destructor.
using DVec = nn::Vec<nn::dyn>;
using DMat = nn::Mat<nn::dyn, nn::dyn>;
static_assert(!std::is_trivially_destructible_v<DVec>);
static_assert(std::is_trivially_copyable_v<nn::Mat<2, 3>>);

namespace model {

using nn::Mat;
using nn::Vec;

// One affine layer and a loss, at any shape (nn::dyn included). `W * x + b`
// calls nn::operator* and nn::operator+, which have no rule: the engine
// inlines them and records the primitives inside (matvec, add).
template <std::size_t O, std::size_t I>
double dense_mse(const Mat<O, I> &W, const Vec<O> &b, const Vec<I> &x,
                 const Vec<O> &t) {
  Vec<O> y = W * x + b;
  return mse(y, t);
}

// A layer written once, used at any shape: a function template, inlined at
// each call.
template <std::size_t O, std::size_t I>
Vec<O> dense_relu(const Mat<O, I> &W, const Vec<O> &b, const Vec<I> &x) {
  Vec<O> z = W * x + b;
  return relu(z);
}

inline constexpr std::size_t H = 8; // hidden width

template <std::size_t In, std::size_t Hid, std::size_t Out>
double mlp_loss(const Mat<Hid, In> &W1, const Vec<Hid> &b1,
                const Mat<Out, Hid> &W2, const Vec<Out> &b2, const Vec<In> &x,
                const Vec<Out> &t) {
  Vec<Hid> h = dense_relu(W1, b1, x);
  Vec<Out> y = sigmoid(W2 * h + b2);
  return mse(y, t);
}

// Scalar C++ around tensor code: built-in arithmetic, literals, a constant
// global, an int converted to double, a value used twice by one primitive
// (dot(W, W)), and a local that does not reach the result.
inline constexpr double weight_decay = 1e-2;

template <std::size_t O, std::size_t I>
double regularised(const Mat<O, I> &W, const Vec<O> &b, const Vec<I> &x,
                   const Vec<O> &t) {
  double data = dense_mse(W, b, x, t);
  double penalty = dot(W, W);
  [[maybe_unused]] double unused = sum(b); // dead: never evaluated
  double four = 4;
  double neg = -penalty;
  return 0.5 * data - weight_decay * neg - data / four;
}

// XOR as a batch of four, the data baked in as constants: each mlp_loss call
// is inlined, so the parameters fan out to four uses whose adjoints add up.
inline constexpr Vec<2> x00{{0.0, 0.0}};
inline constexpr Vec<2> x01{{0.0, 1.0}};
inline constexpr Vec<2> x10{{1.0, 0.0}};
inline constexpr Vec<2> x11{{1.0, 1.0}};
inline constexpr Vec<1> lo{{0.0}};
inline constexpr Vec<1> hi{{1.0}};

template <std::size_t Hid>
double xor_loss(const Mat<Hid, 2> &W1, const Vec<Hid> &b1,
                const Mat<1, Hid> &W2, const Vec<1> &b2) {
  double l00 = mlp_loss(W1, b1, W2, b2, x00, lo);
  double l01 = mlp_loss(W1, b1, W2, b2, x01, hi);
  double l10 = mlp_loss(W1, b1, W2, b2, x10, hi);
  double l11 = mlp_loss(W1, b1, W2, b2, x11, lo);
  return 0.25 * (l00 + l01 + l10 + l11);
}

// Inference: ordinary C++, not reflected.
template <std::size_t In, std::size_t Hid, std::size_t Out>
Vec<Out> predict(const Mat<Hid, In> &W1, const Vec<Hid> &b1,
                 const Mat<Out, Hid> &W2, const Vec<Out> &b2,
                 const Vec<In> &x) {
  return nn::sigmoid(W2 * dense_relu(W1, b1, x) + b2);
}

// Static and dynamic extents mixed: W has dynamic rows and I columns, so W * x
// is dynamic. Two locals of a heap-backed type.
template <std::size_t I>
double mixed_loss(const nn::Mat<nn::dyn, I> &W, const DVec &b,
                  const nn::Vec<I> &x, const DVec &t) {
  DVec h = relu(W * x + b);
  DVec y = sigmoid(hadamard(h, b));
  return mse(y, t);
}

// Elementwise ops on a Vec<N> and a Vec<dyn>: the result is a Vec<N>, and
// each gradient comes back in its parameter's type.
template <std::size_t N>
double mixed_elementwise(const Vec<N> &a, const DVec &b) {
  Vec<N> y = hadamard(a + b, b);
  return dot(y, b) + mse(a, b);
}

// The primitives the MLPs don't use: matmul, outer, sub, scale, transpose,
// sum. transpose(A) is not square, so a wrong transpose rule shows.
template <std::size_t R, std::size_t C>
double matrix_ops(const Mat<R, C> &A, const Mat<C, R> &B, const Vec<R> &u,
                  const Vec<C> &v) {
  Mat<R, R> Q = A * B - outer(u, u);
  Mat<R, C> S = 0.5 * outer(u, v);
  Mat<C, R> T = transpose(A);
  return sum(Q) + dot(S, A) + dot(T, B);
}

// Negation promotes like arithmetic: -c is an int, not an unsigned char.
inline double neg_promoted(double x, unsigned char c) { return -c * x; }

// W.r.t. b, grad() computes neither sigmoid(a) nor its sum: they don't depend
// on b, and dot's pullback reads only a and b.
inline double pruned(const DVec &a, const DVec &b) {
  return dot(a, b) + sum(sigmoid(a));
}

// k::t is the constant, not the parameter t.
namespace k {
inline constexpr double t = 100;
}
inline double qualified_constant(double t) {
  double u = t * 2.0;
  return u * k::t;
}

// <cmath> calls are primitives too, around tensor code.
template <std::size_t N> double cmath_mix(const Vec<N> &w, double s) {
  double e = std::exp(s);
  double l = std::log(dot(w, w));
  return e * l + std::sin(s) * std::cos(s) + std::tanh(sum(w)) + std::sqrt(s);
}

// The float overloads.
inline float cmath_float(float x) { return std::exp(x) * std::sin(x); }

inline double just_exp(double x) { return std::exp(x); }

} // namespace model

// ---------------------------------------------------------------------------
// The reflected DAG, checked at compile time.
// ---------------------------------------------------------------------------

namespace nad = nn::ad;

template <nad::Kind K>
consteval std::size_t count_live(std::span<const nad::Node> tape,
                                 bool varied_only = false) {
  std::size_t n = 0;
  for (const nad::Node &node : tape)
    n += node.kind == K && node.live && (node.varied || !varied_only);
  return n;
}

template <nad::Kind K>
consteval std::size_t count_needed(std::span<const nad::Node> tape) {
  std::size_t n = 0;
  for (const nad::Node &node : tape)
    n += node.kind == K && node.needed;
  return n;
}

// The 2-H-1 MLP of the XOR test.
constexpr auto mlp = ^^model::mlp_loss<2, model::H, 1>;

// Seven primitive calls: matvec, add, relu, matvec, add, sigmoid, mse. The
// operators and dense_relu were inlined, so none of them is a call node.
constexpr auto mlp_all = nad::tape<mlp, nad::mask_of<0, 1, 2, 3>>;
static_assert(count_live<nad::Kind::Call>(mlp_all) == 7);
static_assert(count_live<nad::Kind::Call>(mlp_all, true) == 7);
// The data (x, t) are inputs but not differentiated.
static_assert(mlp_all[0].varied && mlp_all[3].varied);
static_assert(!mlp_all[4].varied && !mlp_all[5].varied);
// The reverse sweep doesn't read the loss, so grad() skips the mse call.
static_assert(count_needed<nad::Kind::Call>(mlp_all) == 6);

// Differentiating the output layer only: the hidden layer's three calls are
// not varied, so the reverse sweep skips them.
constexpr auto mlp_out = nad::tape<mlp, nad::mask_of<2, 3>>;
static_assert(count_live<nad::Kind::Call>(mlp_out, true) == 4);

// dense_mse's matvec, add, mse, then dot; `sum(b)` is dead. (`double four =
// 4` is a Convert node on Clang; GCC folds it to 4.0 before reflection sees
// it.)
constexpr auto reg =
    nad::tape<^^model::regularised<2, 3>, nad::mask_of<0, 1, 2>>;
static_assert(count_live<nad::Kind::Call>(reg) == 4);
static_assert(count_live<nad::Kind::Neg>(reg) == 1);

// Four inlined copies of the MLP.
constexpr auto xor_tape =
    nad::tape<^^model::xor_loss<model::H>, nad::mask_of<0, 1, 2, 3>>;
static_assert(count_live<nad::Kind::Call>(xor_tape) == 4 * 7);

// matvec, add, relu, hadamard, sigmoid, mse.
constexpr auto mixed_tape =
    nad::tape<^^model::mixed_loss<3>, nad::mask_of<0, 1, 2>>;
static_assert(count_live<nad::Kind::Call>(mixed_tape) == 6);

// matmul, outer, sub, outer, scale, transpose, sum, dot, dot.
constexpr auto ops_tape =
    nad::tape<^^model::matrix_ops<2, 3>, nad::mask_of<0, 1, 2, 3>>;
static_assert(count_live<nad::Kind::Call>(ops_tape) == 9);

// dot, sigmoid, sum are live, but grad<1>() computes none of them.
constexpr auto pruned_tape = nad::tape<^^model::pruned, nad::mask_of<1>>;
static_assert(count_live<nad::Kind::Call>(pruned_tape) == 3);
static_assert(count_needed<nad::Kind::Call>(pruned_tape) == 0);

// dot, sum and six <cmath> calls. W.r.t. s, grad() skips tanh(sum(w)): nothing
// varied reads it.
constexpr auto cmath_tape = nad::tape<^^model::cmath_mix<3>, nad::mask_of<1>>;
static_assert(count_live<nad::Kind::Call>(cmath_tape) == 8);
static_assert(count_needed<nad::Kind::Call>(cmath_tape) == 6);

// An engine with only the tensor rules builds the same MLP tape.
static_assert(
    nad::engine<nn::tensor_rule_set>::tape<mlp, nad::mask_of<0, 1, 2, 3>>.size() ==
    mlp_all.size());

// A rule for exp(double) that is deliberately wrong, to see which rule wins.
struct doubled_exp {
  static constexpr std::span<const nad::info> callees =
      nad::probed({^^nad::cmath_probe::exp<double>});
  static constexpr auto d_y(auto y) { return 2 * y; }
};

// ---------------------------------------------------------------------------
// Run time.
// ---------------------------------------------------------------------------

// Deterministic, roughly uniform in [-scale, scale]. `sizes` are T's dynamic
// extents.
template <nn::tensor T, class... I>
T filled_with(double seed, double scale, I... sizes) {
  T t = nn::zeros<T>(sizes...);
  for (std::size_t i = 0; i < t.d.size(); ++i)
    t.d[i] = scale * std::sin(seed + 1.7 * static_cast<double>(i) +
                              0.3 * static_cast<double>(i * i));
  return t;
}

// The reflected gradient `g` of f w.r.t. argument I matches central finite
// differences, element by element.
template <std::size_t I, class F, class Args, class G>
void expect_matches_fd(F f, Args args, const G &g) {
  constexpr double h = 1e-6;
  auto &p = std::get<I>(args);
  for (std::size_t k = 0; k < g.d.size(); ++k) {
    const double orig = p.d[k];
    p.d[k] = orig + h;
    const double up = std::apply(f, args);
    p.d[k] = orig - h;
    const double down = std::apply(f, args);
    p.d[k] = orig;
    EXPECT_NEAR_ABS(g.d[k], (up - down) / (2 * h), 1e-7);
  }
}

using model::H;
using nn::Mat;
using nn::Vec;

void test_dense_mse() {
  const auto W = filled_with<Mat<2, 3>>(0.1, 1.0);
  const auto b = filled_with<Vec<2>>(0.2, 0.5);
  const auto x = filled_with<Vec<3>>(0.3, 1.0);
  const auto t = filled_with<Vec<2>>(0.4, 1.0);

  // Parameters and, for once, the input too (a saliency map).
  auto [L, g] =
      nad::value_and_grad<^^model::dense_mse<2, 3>, 0, 1, 2>(W, b, x, t);
  const auto &[gW, gb, gx] = g;

  EXPECT_EQUAL(L, model::dense_mse(W, b, x, t));
  const auto args = std::tuple{W, b, x, t};
  expect_matches_fd<0>(model::dense_mse<2, 3>, args, gW);
  expect_matches_fd<1>(model::dense_mse<2, 3>, args, gb);
  expect_matches_fd<2>(model::dense_mse<2, 3>, args, gx);

  // In closed form: dL/db = 2 (y - t) / 2.
  const Vec<2> y = W * x + b;
  for (std::size_t i = 0; i < 2; ++i)
    EXPECT_NEAR_ABS(gb[i], y[i] - t[i], 1e-15);
}

void test_mlp() {
  const auto W1 = filled_with<Mat<H, 2>>(1.0, 1.0);
  const auto b1 = filled_with<Vec<H>>(2.0, 0.5);
  const auto W2 = filled_with<Mat<1, H>>(3.0, 1.0);
  const auto b2 = filled_with<Vec<1>>(4.0, 0.5);
  const auto x = filled_with<Vec<2>>(5.0, 1.0);
  const Vec<1> t{{1.0}};

  auto [L, g] = nad::value_and_grad<mlp, 0, 1, 2, 3>(W1, b1, W2, b2, x, t);
  const auto &[gW1, gb1, gW2, gb2] = g;

  constexpr auto f = model::mlp_loss<2, H, 1>;
  EXPECT_EQUAL(L, f(W1, b1, W2, b2, x, t));
  const auto args = std::tuple{W1, b1, W2, b2, x, t};
  expect_matches_fd<0>(f, args, gW1);
  expect_matches_fd<1>(f, args, gb1);
  expect_matches_fd<2>(f, args, gW2);
  expect_matches_fd<3>(f, args, gb2);

  // Fewer parameters, same values for those asked.
  auto [gW2_only] = nad::grad<mlp, 2>(W1, b1, W2, b2, x, t);
  for (std::size_t i = 0; i < H; ++i)
    EXPECT_EQUAL(gW2_only.d[i], gW2.d[i]);

  // grad() skips the loss value but not the gradient.
  const auto [hW1, hb1, hW2, hb2] =
      nad::grad<mlp, 0, 1, 2, 3>(W1, b1, W2, b2, x, t);
  EXPECT_TRUE(hW1.d == gW1.d && hb1.d == gb1.d);
  EXPECT_TRUE(hW2.d == gW2.d && hb2.d == gb2.d);
}

void test_regularised() {
  const auto W = filled_with<Mat<2, 3>>(0.5, 1.0);
  const auto b = filled_with<Vec<2>>(0.6, 0.5);
  const auto x = filled_with<Vec<3>>(0.7, 1.0);
  const auto t = filled_with<Vec<2>>(0.8, 1.0);

  auto [L, g] =
      nad::value_and_grad<^^model::regularised<2, 3>, 0, 1>(W, b, x, t);
  const auto &[gW, gb] = g;

  constexpr auto f = model::regularised<2, 3>;
  EXPECT_NEAR_ABS(L, f(W, b, x, t), 1e-15);
  const auto args = std::tuple{W, b, x, t};
  expect_matches_fd<0>(f, args, gW);
  expect_matches_fd<1>(f, args, gb);
}

void test_dynamic_extents() {
  {
    const auto W = filled_with<DMat>(0.1, 1.0, 2, 3);
    const auto b = filled_with<DVec>(0.2, 0.5, 2);
    const auto x = filled_with<DVec>(0.3, 1.0, 3);
    const auto t = filled_with<DVec>(0.4, 1.0, 2);

    constexpr auto fn = model::dense_mse<nn::dyn, nn::dyn>;
    auto [L, g] =
        nad::value_and_grad<^^model::dense_mse<nn::dyn, nn::dyn>, 0, 1, 2>(
            W, b, x, t);
    const auto &[gW, gb, gx] = g;

    EXPECT_EQUAL(L, fn(W, b, x, t));
    EXPECT_TRUE(gW.extents() == W.extents());
    const auto args = std::tuple{W, b, x, t};
    expect_matches_fd<0>(fn, args, gW);
    expect_matches_fd<1>(fn, args, gb);
    expect_matches_fd<2>(fn, args, gx);
  }

  const auto W = filled_with<nn::Mat<nn::dyn, 3>>(0.9, 1.0, 4);
  const auto b = filled_with<DVec>(1.1, 0.5, 4);
  const auto x = filled_with<Vec<3>>(1.2, 1.0);
  const auto t = filled_with<DVec>(1.3, 1.0, 4);

  auto [L, g] =
      nad::value_and_grad<^^model::mixed_loss<3>, 0, 1, 2>(W, b, x, t);
  const auto &[gW, gb, gx] = g;

  constexpr auto mixed = model::mixed_loss<3>;
  EXPECT_EQUAL(L, mixed(W, b, x, t));
  const auto args = std::tuple{W, b, x, t};
  expect_matches_fd<0>(mixed, args, gW);
  expect_matches_fd<1>(mixed, args, gb);
  expect_matches_fd<2>(mixed, args, gx);

  // Dynamic extents that disagree are a run-time error.
  bool threw = false;
  try {
    (void)model::mixed_loss(W, filled_with<DVec>(1.1, 0.5, 5), x, t);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  EXPECT_TRUE(threw);

  const auto a = filled_with<Vec<3>>(1.4, 1.0);
  const auto c = filled_with<DVec>(1.5, 1.0, 3);
  auto [Le, ge] =
      nad::value_and_grad<^^model::mixed_elementwise<3>, 0, 1>(a, c);
  const auto &[ga, gc] = ge;
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(gc)>, DVec>);

  constexpr auto elementwise = model::mixed_elementwise<3>;
  EXPECT_NEAR_ABS(Le, elementwise(a, c), 1e-15);
  const auto eargs = std::tuple{a, c};
  expect_matches_fd<0>(elementwise, eargs, ga);
  expect_matches_fd<1>(elementwise, eargs, gc);

  // The skipped values are dynamic: grad() still gets every shape it reads.
  auto [gb_pruned] = nad::grad<^^model::pruned, 1>(c, c);
  EXPECT_TRUE(gb_pruned.d == c.d);
  const auto [Lp, gp] = nad::value_and_grad<^^model::pruned, 1>(c, c);
  EXPECT_NEAR_ABS(Lp, model::pruned(c, c), 1e-15);
  EXPECT_TRUE(std::get<0>(gp).d == c.d);
}

void test_matrix_ops() {
  const auto A = filled_with<Mat<2, 3>>(1.6, 1.0);
  const auto B = filled_with<Mat<3, 2>>(1.7, 1.0);
  const auto u = filled_with<Vec<2>>(1.8, 1.0);
  const auto v = filled_with<Vec<3>>(1.9, 1.0);

  auto [L, g] =
      nad::value_and_grad<^^model::matrix_ops<2, 3>, 0, 1, 2, 3>(A, B, u, v);
  const auto &[gA, gB, gu, gv] = g;

  constexpr auto f = model::matrix_ops<2, 3>;
  EXPECT_NEAR_ABS(L, f(A, B, u, v), 1e-14);
  const auto args = std::tuple{A, B, u, v};
  expect_matches_fd<0>(f, args, gA);
  expect_matches_fd<1>(f, args, gB);
  expect_matches_fd<2>(f, args, gu);
  expect_matches_fd<3>(f, args, gv);
}

void test_scalar_semantics() {
  const unsigned char one = 1;
  auto [v1, g1] = nad::value_and_grad<^^model::neg_promoted, 0>(2.0, one);
  EXPECT_EQUAL(v1, -2.0);
  EXPECT_EQUAL(std::get<0>(g1), -1.0);

  auto [v2, g2] = nad::value_and_grad<^^model::qualified_constant, 0>(1.5);
  EXPECT_EQUAL(v2, 300.0);
  EXPECT_EQUAL(std::get<0>(g2), 200.0);
}

void test_cmath() {
  const auto w = filled_with<Vec<3>>(2.1, 1.0);
  const double s = 0.7;
  auto [L, g] = nad::value_and_grad<^^model::cmath_mix<3>, 0, 1>(w, s);
  const auto &[gw, gs] = g;

  constexpr auto f = model::cmath_mix<3>;
  EXPECT_EQUAL(L, f(w, s));
  expect_matches_fd<0>(f, std::tuple{w, s}, gw);
  constexpr double h = 1e-6;
  EXPECT_NEAR_ABS(gs, (f(w, s + h) - f(w, s - h)) / (2 * h), 1e-7);

  auto [gs_only] = nad::grad<^^model::cmath_mix<3>, 1>(w, s);
  EXPECT_EQUAL(gs_only, gs);

  const float x = 0.5f;
  auto [vf, gf] = nad::value_and_grad<^^model::cmath_float, 0>(x);
  static_assert(std::is_same_v<decltype(vf), float>);
  EXPECT_EQUAL(vf, model::cmath_float(x));
  EXPECT_NEAR_ABS(std::get<0>(gf), std::exp(x) * (std::sin(x) + std::cos(x)),
                  1e-6);
}

// The first rule listing a callee wins.
void test_rule_precedence() {
  using override_first = nad::engine<doubled_exp, nad::default_rules>;
  using override_last = nad::engine<nad::default_rules, doubled_exp>;
  auto [g_first] = override_first::grad<^^model::just_exp, 0>(0.5);
  auto [g_last] = override_last::grad<^^model::just_exp, 0>(0.5);
  EXPECT_EQUAL(g_first, 2 * std::exp(0.5));
  EXPECT_EQUAL(g_last, std::exp(0.5));
}

void test_xor_training() {
  auto W1 = filled_with<Mat<H, 2>>(1.0, 1.0);
  auto b1 = nn::filled<Vec<H>>(0.1);
  auto W2 = filled_with<Mat<1, H>>(2.0, 0.5);
  auto b2 = Vec<1>{};

  // The batch gradient: four inlined MLPs sharing the parameters.
  constexpr auto loss = model::xor_loss<H>;
  {
    auto [L, g] =
        nad::value_and_grad<^^model::xor_loss<H>, 0, 1, 2, 3>(W1, b1, W2, b2);
    const auto &[gW1, gb1, gW2, gb2] = g;
    EXPECT_EQUAL(L, loss(W1, b1, W2, b2));
    const auto args = std::tuple{W1, b1, W2, b2};
    expect_matches_fd<0>(loss, args, gW1);
    expect_matches_fd<1>(loss, args, gb1);
    expect_matches_fd<2>(loss, args, gW2);
    expect_matches_fd<3>(loss, args, gb2);
  }

  // Full-batch gradient descent.
  const double initial = loss(W1, b1, W2, b2);
  constexpr double lr = 2.0;
  for (int step = 0; step < 2000; ++step) {
    auto [gW1, gb1, gW2, gb2] =
        nad::grad<^^model::xor_loss<H>, 0, 1, 2, 3>(W1, b1, W2, b2);
    nn::sgd_update(W1, gW1, lr);
    nn::sgd_update(b1, gb1, lr);
    nn::sgd_update(W2, gW2, lr);
    nn::sgd_update(b2, gb2, lr);
  }
  const double final_loss = loss(W1, b1, W2, b2);
  std::cout << "XOR loss: " << initial << " -> " << final_loss << '\n';

  EXPECT_LESS_THAN(final_loss, 0.01);
  EXPECT_LESS_THAN(final_loss, initial / 10);
  EXPECT_LESS_THAN(model::predict(W1, b1, W2, b2, model::x00)[0], 0.5);
  EXPECT_LESS_THAN(0.5, model::predict(W1, b1, W2, b2, model::x01)[0]);
  EXPECT_LESS_THAN(0.5, model::predict(W1, b1, W2, b2, model::x10)[0]);
  EXPECT_LESS_THAN(model::predict(W1, b1, W2, b2, model::x11)[0], 0.5);
}

int main() {
  test_dense_mse();
  test_mlp();
  test_regularised();
  test_dynamic_extents();
  test_matrix_ops();
  test_scalar_semantics();
  test_cmath();
  test_rule_precedence();
  test_xor_training();
  TEST_END;
}
