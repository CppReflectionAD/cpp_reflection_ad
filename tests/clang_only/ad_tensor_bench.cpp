// TEST-FLAGS: -O2
// ad_tensor_bench.cpp — tensor/ML AD: reflection reverse-mode vs a runtime tape
// vs hand-written, on the same MLP gradient.
//
// The runtime tape is the "how ML AD is usually done" baseline: a Var type that
// records the computation graph at run time, then a backward pass. Reflection
// generates the same reverse sweep at compile time (no graph object, no dynamic
// dispatch, dead work pruned). Hand-written is the floor.
//
// We report absolute time as well as ratios: for tensors the *ratio* over hand-
// written is smaller than for scalars (matmuls dominate), but the *absolute*
// time saved per call is larger.
//
// Build/run: `make run-tensor-bench` (built at -O2).

#include "autograd_tensor.h"
#include "tensor.h"
#include "tensor_rules.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define NOINLINE __attribute__((noinline))
template <class T> static inline void do_not_optimize(T &v) { asm volatile("" : "+r,m"(v) : : "memory"); }

using nn::Tensor;
using Grad4 = std::array<Tensor, 4>;

// reduce g to shape (r,c) — reverse of broadcasting, by target shape
static Tensor unbroadcast_to(const Tensor &g, int r, int c) {
  Tensor o(r, c);
  for (int i = 0; i < g.r; ++i)
    for (int j = 0; j < g.c; ++j) o.at(i % r, j % c) += g.at(i, j);
  return o;
}

// ---------------------------------------------------------------------------
// Baseline 1: runtime reverse-mode tape (a mini eager "autograd").
// ---------------------------------------------------------------------------
namespace rt {
enum class TOp { Leaf, Matmul, Add, Mul, Relu, Sum };
struct Entry { TOp op; int a, b; int ar, ac, br, bc; Tensor sa, sb; };  // saved for backward
static std::vector<Entry> tape;   // reused across calls (clear keeps capacity)

struct Var { Tensor v; int id; };
static Var leaf(Tensor t) {
  int id = (int)tape.size();
  tape.push_back({TOp::Leaf, -1, -1, t.r, t.c, 0, 0, {}, {}});
  return {std::move(t), id};
}
static Var push(TOp op, const Var &A, const Var &B, Tensor res, Tensor sa, Tensor sb) {
  int id = (int)tape.size();
  tape.push_back({op, A.id, B.id, A.v.r, A.v.c, B.v.r, B.v.c, std::move(sa), std::move(sb)});
  return {std::move(res), id};
}
static Var push1(TOp op, const Var &A, Tensor res, Tensor sa) {
  int id = (int)tape.size();
  tape.push_back({op, A.id, -1, A.v.r, A.v.c, 0, 0, std::move(sa), {}});
  return {std::move(res), id};
}
static Var matmul(const Var &A, const Var &B) { return push(TOp::Matmul, A, B, nn::matmul(A.v, B.v), A.v, B.v); }
static Var add(const Var &A, const Var &B)    { return push(TOp::Add, A, B, nn::add(A.v, B.v), {}, {}); }
static Var mul(const Var &A, const Var &B)    { return push(TOp::Mul, A, B, nn::mul(A.v, B.v), A.v, B.v); }
static Var relu(const Var &A)                 { return push1(TOp::Relu, A, nn::relu(A.v), A.v); }
static Var sum(const Var &A)                  { return push1(TOp::Sum, A, nn::sum(A.v), A.v); }

NOINLINE Grad4 grad(const Tensor &x, const Tensor &W1, const Tensor &b1,
                    const Tensor &W2, const Tensor &b2) {
  tape.clear();
  Var vx = leaf(x), vW1 = leaf(W1), vb1 = leaf(b1), vW2 = leaf(W2), vb2 = leaf(b2);
  Var z1 = matmul(vx, vW1), a1 = add(z1, vb1), h = relu(a1);
  Var z2 = matmul(h, vW2), o = add(z2, vb2), sq = mul(o, o), L = sum(sq);

  std::vector<Tensor> adj(tape.size());
  std::vector<char> seen(tape.size(), 0);
  auto acc = [&](int i, Tensor g) {
    if (seen[i]) adj[i] = nn::add(adj[i], g); else { adj[i] = std::move(g); seen[i] = 1; }
  };
  adj[L.id] = nn::ones_like(L.v); seen[L.id] = 1;
  for (int i = (int)tape.size() - 1; i >= 0; --i) {
    if (!seen[i]) continue;
    Entry &e = tape[i]; Tensor g = adj[i];
    switch (e.op) {
      case TOp::Matmul: acc(e.a, nn::matmul(g, nn::transpose(e.sb)));
                        acc(e.b, nn::matmul(nn::transpose(e.sa), g)); break;
      case TOp::Add:    acc(e.a, unbroadcast_to(g, e.ar, e.ac));
                        acc(e.b, unbroadcast_to(g, e.br, e.bc)); break;
      case TOp::Mul:    acc(e.a, unbroadcast_to(nn::mul(g, e.sb), e.ar, e.ac));
                        acc(e.b, unbroadcast_to(nn::mul(g, e.sa), e.br, e.bc)); break;
      case TOp::Relu:   acc(e.a, nn::mul(g, nn::d_relu(e.sa))); break;
      case TOp::Sum:    acc(e.a, nn::mul(nn::ones_like(e.sa), g)); break;
      case TOp::Leaf:   break;
    }
  }
  return {adj[vW1.id], adj[vb1.id], adj[vW2.id], adj[vb2.id]};
}
}  // namespace rt

// ---------------------------------------------------------------------------
// Baseline 2: hand-written gradient (the floor).
// ---------------------------------------------------------------------------
NOINLINE Grad4 grad_hand(const Tensor &x, const Tensor &W1, const Tensor &b1,
                         const Tensor &W2, const Tensor &b2) {
  using namespace nn;
  Tensor z1 = matmul(x, W1), a1 = add(z1, b1), h = relu(a1);
  Tensor z2 = matmul(h, W2), o = add(z2, b2);
  Tensor d_o  = mul(o, Tensor(2.0));                 // d/do sum(o*o) = 2o
  Tensor d_b2 = unbroadcast(d_o, b2);
  Tensor d_W2 = matmul(transpose(h), d_o);
  Tensor d_h  = matmul(d_o, transpose(W2));
  Tensor d_a1 = mul(d_h, d_relu(a1));
  Tensor d_b1 = unbroadcast(d_a1, b1);
  Tensor d_W1 = matmul(transpose(x), d_a1);
  return {d_W1, d_b1, d_W2, d_b2};
}

// Baseline 3: reflection reverse-mode.
NOINLINE Grad4 grad_reflect(const Tensor &x, const Tensor &W1, const Tensor &b1,
                            const Tensor &W2, const Tensor &b2) {
  return ad::gradient_wrt<^^nn::loss, Tensor, 1, 2, 3, 4>(x, W1, b1, W2, b2);
}

// ---------------------------------------------------------------------------
using clk = std::chrono::steady_clock;
static bool close4(const Grad4 &A, const Grad4 &B) {
  for (int k = 0; k < 4; ++k)
    for (std::size_t i = 0; i < A[k].d.size(); ++i)
      if (std::abs(A[k].d[i] - B[k].d[i]) > 1e-6 * (1 + std::abs(B[k].d[i]))) return false;
  return true;
}

template <class F>
static double bench(F fn, Tensor &x, const Tensor &W1, const Tensor &b1,
                    const Tensor &W2, const Tensor &b2, std::size_t iters, int trials) {
  double best = 1e300;
  for (int t = 0; t < trials; ++t) {
    double sink = 0.0;
    auto t0 = clk::now();
    for (std::size_t i = 0; i < iters; ++i) {
      x.d[0] = 0.5 + (i & 1023) * 1e-6;                 // vary input (defeat hoisting)
      Grad4 g = fn(x, W1, b1, W2, b2);
      sink += g[0].d[0] + g[1].d[0] + g[2].d[0] + g[3].d[0];
      do_not_optimize(sink);
    }
    auto t1 = clk::now();
    do_not_optimize(sink);
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    if (ns < best) best = ns;
  }
  return best;
}

int main(int argc, char **argv) {
  std::size_t iters = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 30000;
  bool csv = (argc > 2) && std::string(argv[2]) == "--csv";
  int trials = 5;

  int B = 32, In = 32, H = 64, Out = 16;
  Tensor x(B, In), W1(In, H), b1(1, H), W2(H, Out), b2(1, Out);
  auto fill = [](Tensor &t, double s) {
    for (std::size_t i = 0; i < t.d.size(); ++i) t.d[i] = std::sin(s + 0.13 * i) * 0.3;
  };
  fill(x, 0.1); fill(W1, 1.0); fill(b1, 2.0); fill(W2, 3.0); fill(b2, 4.0);

  if (!(close4(grad_reflect(x, W1, b1, W2, b2), grad_hand(x, W1, b1, W2, b2)) &&
        close4(rt::grad(x, W1, b1, W2, b2), grad_hand(x, W1, b1, W2, b2)))) {
    std::printf("CORRECTNESS FAILED\n"); return 1;
  }

  double t_hand = bench(grad_hand,    x, W1, b1, W2, b2, iters, trials);
  double t_refl = bench(grad_reflect, x, W1, b1, W2, b2, iters, trials);
  double t_tape = bench(rt::grad,     x, W1, b1, W2, b2, iters, trials);

  if (csv) {
    std::printf("group,method,ns\n");
    std::printf("tensor,hand,%.3f\n", t_hand);
    std::printf("tensor,reflection_reverse,%.3f\n", t_refl);
    std::printf("tensor,runtime_tape,%.3f\n", t_tape);
    return 0;
  }
  std::printf("=== MLP gradient (B=%d In=%d H=%d Out=%d), min ns/call over %d trials ===\n",
              B, In, H, Out, trials);
  std::printf("  hand-written        %9.1f ns   %.2fx\n", t_hand, 1.0);
  std::printf("  reflection reverse  %9.1f ns   %.2fx\n", t_refl, t_refl / t_hand);
  std::printf("  runtime tape        %9.1f ns   %.2fx\n", t_tape, t_tape / t_hand);
  std::printf("\nnet time saved vs runtime tape: %.0f ns/call  (reflection is %.2fx faster)\n",
              t_tape - t_refl, t_tape / t_refl);
  return 0;
}
