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
#include <tuple>
#include <type_traits>
#include <vector>

// A tensor with heap storage, so with a non-trivial destructor: GCC wraps
// each temporary of it in a cleanup, which the engine sees through. In nn,
// like Vec, so that nn's operators are found for it, in user code and in the
// code the engine generates.
namespace nn {
template <std::size_t N> struct HeapVec {
  static constexpr std::size_t size = N;
  std::vector<double> d = std::vector<double>(N);
};
static_assert(!std::is_trivially_destructible_v<HeapVec<3>>);
} // namespace nn

namespace model {

using nn::Mat;
using nn::Vec;

// One affine layer and a loss. `W * x + b` calls nn::operator* and
// nn::operator+, which have no rule: the engine inlines them and records the
// primitives inside (matvec, add).
inline double dense_mse(const Mat<2, 3> &W, const Vec<2> &b, const Vec<3> &x,
                        const Vec<2> &t) {
  Vec<2> y = W * x + b;
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

inline double mlp_loss(const Mat<H, 2> &W1, const Vec<H> &b1,
                       const Mat<1, H> &W2, const Vec<1> &b2, const Vec<2> &x,
                       const Vec<1> &t) {
  Vec<H> h = dense_relu(W1, b1, x);
  Vec<1> y = sigmoid(W2 * h + b2);
  return mse(y, t);
}

// Scalar C++ around tensor code: built-in arithmetic, literals, a constant
// global, an int converted to double, a value used twice by one primitive
// (dot(W, W)), and a local that does not reach the result.
inline constexpr double weight_decay = 1e-2;

inline double regularised(const Mat<2, 3> &W, const Vec<2> &b, const Vec<3> &x,
                          const Vec<2> &t) {
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

inline double xor_loss(const Mat<H, 2> &W1, const Vec<H> &b1,
                       const Mat<1, H> &W2, const Vec<1> &b2) {
  double l00 = mlp_loss(W1, b1, W2, b2, x00, lo);
  double l01 = mlp_loss(W1, b1, W2, b2, x01, hi);
  double l10 = mlp_loss(W1, b1, W2, b2, x10, hi);
  double l11 = mlp_loss(W1, b1, W2, b2, x11, lo);
  return 0.25 * (l00 + l01 + l10 + l11);
}

// Inference: ordinary C++, not reflected.
inline double predict(const Mat<H, 2> &W1, const Vec<H> &b1,
                      const Mat<1, H> &W2, const Vec<1> &b2, const Vec<2> &x) {
  return nn::sigmoid(W2 * dense_relu(W1, b1, x) + b2)[0];
}

// Two locals of the heap-backed type.
inline double heap_loss(const nn::HeapVec<3> &a, const nn::HeapVec<3> &b,
                        const nn::HeapVec<3> &t) {
  nn::HeapVec<3> h = relu(a + b);
  nn::HeapVec<3> y = sigmoid(hadamard(h, a));
  return mse(y, t);
}

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

// Seven primitive calls: matvec, add, relu, matvec, add, sigmoid, mse. The
// operators and dense_relu were inlined, so none of them is a call node.
constexpr auto mlp_all = nad::tape<^^model::mlp_loss, nad::mask_of<0, 1, 2, 3>>;
static_assert(count_live<nad::Kind::Call>(mlp_all) == 7);
static_assert(count_live<nad::Kind::Call>(mlp_all, true) == 7);
// The data (x, t) are inputs but not differentiated.
static_assert(mlp_all[0].varied && mlp_all[3].varied);
static_assert(!mlp_all[4].varied && !mlp_all[5].varied);

// Differentiating the output layer only: the hidden layer's three calls are
// not varied, so the reverse sweep skips them.
constexpr auto mlp_out = nad::tape<^^model::mlp_loss, nad::mask_of<2, 3>>;
static_assert(count_live<nad::Kind::Call>(mlp_out, true) == 4);

// dense_mse's matvec, add, mse, then dot; `sum(b)` is dead. (`double four =
// 4` is a Convert node on Clang; GCC folds it to 4.0 before reflection sees
// it.)
constexpr auto reg = nad::tape<^^model::regularised, nad::mask_of<0, 1, 2>>;
static_assert(count_live<nad::Kind::Call>(reg) == 4);
static_assert(count_live<nad::Kind::Neg>(reg) == 1);

// Four inlined copies of the MLP.
constexpr auto xor_tape =
    nad::tape<^^model::xor_loss, nad::mask_of<0, 1, 2, 3>>;
static_assert(count_live<nad::Kind::Call>(xor_tape) == 4 * 7);

// add (inlined from operator+), relu, hadamard, sigmoid, mse.
constexpr auto heap_tape = nad::tape<^^model::heap_loss, nad::mask_of<0, 1>>;
static_assert(count_live<nad::Kind::Call>(heap_tape) == 5);

// ---------------------------------------------------------------------------
// Run time.
// ---------------------------------------------------------------------------

// Deterministic, roughly uniform in [-scale, scale].
template <nn::tensor T> T filled_with(double seed, double scale) {
  T t;
  for (std::size_t i = 0; i < T::size; ++i)
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
  for (std::size_t k = 0; k < G::size; ++k) {
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
  auto [L, g] = nad::value_and_grad<^^model::dense_mse, 0, 1, 2>(W, b, x, t);
  const auto &[gW, gb, gx] = g;

  EXPECT_EQUAL(L, model::dense_mse(W, b, x, t));
  const auto args = std::tuple{W, b, x, t};
  expect_matches_fd<0>(model::dense_mse, args, gW);
  expect_matches_fd<1>(model::dense_mse, args, gb);
  expect_matches_fd<2>(model::dense_mse, args, gx);

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

  auto [L, g] =
      nad::value_and_grad<^^model::mlp_loss, 0, 1, 2, 3>(W1, b1, W2, b2, x, t);
  const auto &[gW1, gb1, gW2, gb2] = g;

  EXPECT_EQUAL(L, model::mlp_loss(W1, b1, W2, b2, x, t));
  const auto args = std::tuple{W1, b1, W2, b2, x, t};
  expect_matches_fd<0>(model::mlp_loss, args, gW1);
  expect_matches_fd<1>(model::mlp_loss, args, gb1);
  expect_matches_fd<2>(model::mlp_loss, args, gW2);
  expect_matches_fd<3>(model::mlp_loss, args, gb2);

  // Fewer parameters, same values for those asked.
  auto [gW2_only] = nad::grad<^^model::mlp_loss, 2>(W1, b1, W2, b2, x, t);
  for (std::size_t i = 0; i < H; ++i)
    EXPECT_EQUAL(gW2_only.d[i], gW2.d[i]);
}

void test_regularised() {
  const auto W = filled_with<Mat<2, 3>>(0.5, 1.0);
  const auto b = filled_with<Vec<2>>(0.6, 0.5);
  const auto x = filled_with<Vec<3>>(0.7, 1.0);
  const auto t = filled_with<Vec<2>>(0.8, 1.0);

  auto [L, g] = nad::value_and_grad<^^model::regularised, 0, 1>(W, b, x, t);
  const auto &[gW, gb] = g;

  EXPECT_NEAR_ABS(L, model::regularised(W, b, x, t), 1e-15);
  const auto args = std::tuple{W, b, x, t};
  expect_matches_fd<0>(model::regularised, args, gW);
  expect_matches_fd<1>(model::regularised, args, gb);
}

void test_heap_storage() {
  using V = nn::HeapVec<3>;
  const auto a = filled_with<V>(0.9, 1.0);
  const auto b = filled_with<V>(1.1, 0.5);
  const auto t = filled_with<V>(1.3, 1.0);

  auto [L, g] = nad::value_and_grad<^^model::heap_loss, 0, 1>(a, b, t);
  const auto &[ga, gb] = g;

  EXPECT_EQUAL(L, model::heap_loss(a, b, t));
  const auto args = std::tuple{a, b, t};
  expect_matches_fd<0>(model::heap_loss, args, ga);
  expect_matches_fd<1>(model::heap_loss, args, gb);
}

void test_xor_training() {
  auto W1 = filled_with<Mat<H, 2>>(1.0, 1.0);
  auto b1 = nn::filled<Vec<H>>(0.1);
  auto W2 = filled_with<Mat<1, H>>(2.0, 0.5);
  auto b2 = Vec<1>{};

  // The batch gradient: four inlined MLPs sharing the parameters.
  {
    auto [L, g] =
        nad::value_and_grad<^^model::xor_loss, 0, 1, 2, 3>(W1, b1, W2, b2);
    const auto &[gW1, gb1, gW2, gb2] = g;
    EXPECT_EQUAL(L, model::xor_loss(W1, b1, W2, b2));
    const auto args = std::tuple{W1, b1, W2, b2};
    expect_matches_fd<0>(model::xor_loss, args, gW1);
    expect_matches_fd<1>(model::xor_loss, args, gb1);
    expect_matches_fd<2>(model::xor_loss, args, gW2);
    expect_matches_fd<3>(model::xor_loss, args, gb2);
  }

  // Full-batch gradient descent.
  const double initial = model::xor_loss(W1, b1, W2, b2);
  constexpr double lr = 2.0;
  for (int step = 0; step < 2000; ++step) {
    auto [gW1, gb1, gW2, gb2] =
        nad::grad<^^model::xor_loss, 0, 1, 2, 3>(W1, b1, W2, b2);
    nn::sgd_update(W1, gW1, lr);
    nn::sgd_update(b1, gb1, lr);
    nn::sgd_update(W2, gW2, lr);
    nn::sgd_update(b2, gb2, lr);
  }
  const double final_loss = model::xor_loss(W1, b1, W2, b2);
  std::cout << "XOR loss: " << initial << " -> " << final_loss << '\n';

  EXPECT_LESS_THAN(final_loss, 0.01);
  EXPECT_LESS_THAN(final_loss, initial / 10);
  EXPECT_LESS_THAN(model::predict(W1, b1, W2, b2, model::x00), 0.5);
  EXPECT_LESS_THAN(0.5, model::predict(W1, b1, W2, b2, model::x01));
  EXPECT_LESS_THAN(0.5, model::predict(W1, b1, W2, b2, model::x10));
  EXPECT_LESS_THAN(model::predict(W1, b1, W2, b2, model::x11), 0.5);
}

int main() {
  test_dense_mse();
  test_mlp();
  test_regularised();
  test_heap_storage();
  test_xor_training();
  TEST_END;
}
