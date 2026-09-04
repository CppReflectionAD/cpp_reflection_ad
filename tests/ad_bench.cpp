// TEST-FLAGS: -O2
// ad_bench.cpp — performance: reflection AD vs the ways people do AD *without*
// reflection, vs hand-written.
//
// Approaches compared:
//   * hand-written          — the ideal baseline (derivative coded by hand).
//   * reflection AD (ours)  — derivative generated at compile time, spliced.
//   * forward dual numbers  — operator overloading on a Dual{value,deriv} type
//                             (requires making the function a template).
//   * runtime tape (reverse)— a Var type records a graph at run time, then a
//                             backward pass (mini "autograd"; requires rewriting
//                             the function in terms of Var).
//   (source generation, e.g. gccad, emits derivative *source* and compiles it,
//    so its run-time cost equals hand-written — same as ours, different workflow.)
//
// Timing is in-process (the derivative is inlined arithmetic; whole-process
// timing would be startup noise). We defeat the optimizer with a sink + barrier,
// vary the input each iteration, keep each variant noinline so identical ones
// aren't merged, warm up, and take the min over trials. Every AD result is
// checked against hand-written before timing.
//
// Build at -O2 (see `make run-bench`).

#include "autograd.h"
#include "forward_derivative.h"
#include "reverse_derivative.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define NOINLINE __attribute__((noinline))

template <class T> static inline void do_not_optimize(T &v) {
  asm volatile("" : "+r,m"(v) : : "memory");
}

// ===========================================================================
// FORWARD-MODE target: heavy single-input f(x).
// Three sources with identical math: concrete double (for reflection + hand),
// and a template (for dual numbers).
// ===========================================================================
constexpr double heavy_f(double x) {
  double a = std::sin(x) * x + 2.0 * x;
  double b = a * a + std::cos(a);
  double c = std::exp(b * 0.01) + b;
  double d = c * a - std::sin(c);
  double e = d * d + std::log(c * c + 1.0);
  double g = std::cos(e * 0.5) + e * d;
  return g * a + std::sin(g);
}

// Hand-written derivative (forward mode carried by hand as value/deriv pairs).
NOINLINE double dheavy_f_hand(double x) {
  double a = std::sin(x) * x + 2.0 * x,      da = std::cos(x) * x + std::sin(x) + 2.0;
  double b = a * a + std::cos(a),            db = 2.0 * a * da - std::sin(a) * da;
  double c = std::exp(b * 0.01) + b,         dc = std::exp(b * 0.01) * 0.01 * db + db;
  double d = c * a - std::sin(c),            dd = dc * a + c * da - std::cos(c) * dc;
  double e = d * d + std::log(c * c + 1.0),  de = 2.0 * d * dd + (2.0 * c * dc) / (c * c + 1.0);
  double g = std::cos(e * 0.5) + e * d,      dg = -std::sin(e * 0.5) * 0.5 * de + de * d + e * dd;
  return dg * a + g * da + std::cos(g) * dg;
}

NOINLINE double dheavy_f_reflect(double x) {
  return ad::forward_derivative<^^heavy_f, 0>(x);
}

// --- forward-mode dual numbers (the classic no-reflection forward AD) ---
struct Dual {
  double v = 0, d = 0;
  Dual() = default;
  Dual(double c) : v(c), d(0) {}          // literals promote to a constant dual
  Dual(double v_, double d_) : v(v_), d(d_) {}
};
inline Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
inline Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
inline Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
inline Dual operator/(Dual a, Dual b) { return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)}; }
inline Dual operator-(Dual a) { return {-a.v, -a.d}; }
inline Dual sin(Dual a) { return {std::sin(a.v), std::cos(a.v) * a.d}; }
inline Dual cos(Dual a) { return {std::cos(a.v), -std::sin(a.v) * a.d}; }
inline Dual exp(Dual a) { return {std::exp(a.v), std::exp(a.v) * a.d}; }
inline Dual log(Dual a) { return {std::log(a.v), a.d / a.v}; }

template <class T> T heavy_f_t(T x) {         // same math as heavy_f, generic
  using std::sin; using std::cos; using std::exp; using std::log;
  T a = sin(x) * x + 2.0 * x;
  T b = a * a + cos(a);
  T c = exp(b * 0.01) + b;
  T d = c * a - sin(c);
  T e = d * d + log(c * c + 1.0);
  T g = cos(e * 0.5) + e * d;
  return g * a + sin(g);
}
NOINLINE double dheavy_f_dual(double x) { return heavy_f_t(Dual(x, 1.0)).d; }

// ===========================================================================
// REVERSE-MODE target: heavy 4-input scalar g(a,b,c,d).
// ===========================================================================
constexpr double heavy_g(double a, double b, double c, double d) {
  double p = a * b;
  double q = c * d;
  double s = std::sin(p) + std::exp(q * 0.1);
  double t = s * s + a * c + b * d;
  double u = std::cos(t * 0.3) + t * q;
  return u * p + std::log(s * s + 1.0);
}
using Grad4 = std::array<double, 4>;

// Hand-written gradient (reverse by hand, intermediates shared).
NOINLINE Grad4 grad_heavy_g_hand(double a, double b, double c, double d) {
  double p = a * b, q = c * d;
  double ex = std::exp(q * 0.1);
  double s = std::sin(p) + ex;
  double t = s * s + a * c + b * d;
  double u = std::cos(t * 0.3) + t * q;
  // r = u*p + log(s*s + 1)
  double dr_du = p, dr_dp = u;
  double dr_ds = (2.0 * s) / (s * s + 1.0);
  double du_dt = -std::sin(t * 0.3) * 0.3 + q, du_dq = t;
  double dt_ds = 2.0 * s, ds_dp = std::cos(p), ds_dq = ex * 0.1;
  double adj_s = dr_ds + dr_du * du_dt * dt_ds;
  double adj_t = dr_du * du_dt;
  double adj_p = dr_dp + adj_s * ds_dp;
  double adj_q = dr_du * du_dq + adj_s * ds_dq;
  double ga = adj_p * b + adj_t * c;
  double gb = adj_p * a + adj_t * d;
  double gc = adj_q * d + adj_t * a;
  double gd = adj_q * c + adj_t * b;
  return {ga, gb, gc, gd};
}

NOINLINE Grad4 grad_heavy_g_reflect(double a, double b, double c, double d) {
  return ad::gradient_reverse<^^heavy_g>(a, b, c, d);
}
NOINLINE Grad4 grad_heavy_g_reflect_fwd(double a, double b, double c, double d) {
  return ad::gradient_of<^^heavy_g>(a, b, c, d);   // forward mode, P passes
}

// --- runtime reverse-mode tape (the classic no-reflection reverse AD) ---
struct TNode { int p0 = -1, p1 = -1; double w0 = 0, w1 = 0; };
static std::vector<TNode> g_tape;      // reused across calls (clear, keep capacity)
static std::vector<double> g_adj;

struct Var {
  double v; int idx;
  Var(double x) : v(x), idx((int)g_tape.size()) { g_tape.push_back(TNode{}); }  // leaf
  Var(double v_, int i) : v(v_), idx(i) {}
};
static Var mk(double v, int p0, double w0, int p1, double w1) {
  int i = (int)g_tape.size();
  g_tape.push_back(TNode{p0, p1, w0, w1});
  return Var(v, i);
}
inline Var operator+(Var a, Var b) { return mk(a.v + b.v, a.idx, 1.0, b.idx, 1.0); }
inline Var operator-(Var a, Var b) { return mk(a.v - b.v, a.idx, 1.0, b.idx, -1.0); }
inline Var operator*(Var a, Var b) { return mk(a.v * b.v, a.idx, b.v, b.idx, a.v); }
inline Var operator/(Var a, Var b) { return mk(a.v / b.v, a.idx, 1.0 / b.v, b.idx, -a.v / (b.v * b.v)); }
inline Var operator-(Var a) { return mk(-a.v, a.idx, -1.0, -1, 0.0); }
inline Var sin(Var a) { return mk(std::sin(a.v), a.idx, std::cos(a.v), -1, 0.0); }
inline Var cos(Var a) { return mk(std::cos(a.v), a.idx, -std::sin(a.v), -1, 0.0); }
inline Var exp(Var a) { return mk(std::exp(a.v), a.idx, std::exp(a.v), -1, 0.0); }
inline Var log(Var a) { return mk(std::log(a.v), a.idx, 1.0 / a.v, -1, 0.0); }

template <class T> T heavy_g_t(T a, T b, T c, T d) {   // same math as heavy_g
  using std::sin; using std::cos; using std::exp; using std::log;
  T p = a * b;
  T q = c * d;
  T s = sin(p) + exp(q * 0.1);
  T t = s * s + a * c + b * d;
  T u = cos(t * 0.3) + t * q;
  return u * p + log(s * s + 1.0);
}

NOINLINE Grad4 grad_heavy_g_tape(double a, double b, double c, double d) {
  g_tape.clear();
  Var va(a), vb(b), vc(c), vd(d);           // leaves
  Var r = heavy_g_t(va, vb, vc, vd);        // records the graph
  g_adj.assign(g_tape.size(), 0.0);
  g_adj[r.idx] = 1.0;
  for (int i = (int)g_tape.size() - 1; i >= 0; --i) {   // reverse sweep
    double ai = g_adj[i];
    if (ai == 0.0) continue;
    const TNode &n = g_tape[i];
    if (n.p0 >= 0) g_adj[n.p0] += n.w0 * ai;
    if (n.p1 >= 0) g_adj[n.p1] += n.w1 * ai;
  }
  return {g_adj[va.idx], g_adj[vb.idx], g_adj[vc.idx], g_adj[vd.idx]};
}

// ===========================================================================
// Harness
// ===========================================================================
using clk = std::chrono::steady_clock;
static bool approx(double x, double y) { return std::abs(x - y) <= 1e-6 * (1.0 + std::abs(y)); }

template <class F> static double bench_scalar(F fn, std::size_t iters, int trials) {
  double best = 1e300;
  for (int t = 0; t < trials; ++t) {
    double sink = 0.0;
    auto t0 = clk::now();
    for (std::size_t i = 0; i < iters; ++i) {
      double x = 0.3 + (i & 4095) * 1e-4;
      double r = fn(x); sink += r; do_not_optimize(sink);
    }
    auto t1 = clk::now();
    do_not_optimize(sink);
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    if (ns < best) best = ns;
  }
  return best;
}
template <class F> static double bench_grad(F fn, std::size_t iters, int trials) {
  double best = 1e300;
  for (int t = 0; t < trials; ++t) {
    double sink = 0.0;
    auto t0 = clk::now();
    for (std::size_t i = 0; i < iters; ++i) {
      double x = 0.3 + (i & 4095) * 1e-4;
      Grad4 g = fn(x, x * 1.1 + 0.2, x * 0.7 + 0.5, x * 1.3 + 0.1);
      sink += g[0] + g[1] + g[2] + g[3]; do_not_optimize(sink);
    }
    auto t1 = clk::now();
    do_not_optimize(sink);
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    if (ns < best) best = ns;
  }
  return best;
}

int main(int argc, char **argv) {
  std::size_t iters = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 20'000'000;
  std::size_t gi = iters / 4;   // gradient loop does more work; fewer iters
  int trials = 5;
  bool csv = (argc > 2) && std::string(argv[2]) == "--csv";

  // correctness
  {
    double x = 0.9;
    bool ok = approx(dheavy_f_reflect(x), dheavy_f_hand(x)) &&
              approx(dheavy_f_dual(x), dheavy_f_hand(x));
    double a=0.9,b=1.2,c=0.4,d=1.7;
    Grad4 gh=grad_heavy_g_hand(a,b,c,d), gr=grad_heavy_g_reflect(a,b,c,d),
          gt=grad_heavy_g_tape(a,b,c,d), gfw=grad_heavy_g_reflect_fwd(a,b,c,d);
    for (int i=0;i<4;i++) ok = ok && approx(gr[i],gh[i]) && approx(gt[i],gh[i]) && approx(gfw[i],gh[i]);
    if (!ok) { std::printf("CORRECTNESS FAILED\n"); return 1; }
  }

  double f_hand = bench_scalar(dheavy_f_hand,    iters, trials);
  double f_refl = bench_scalar(dheavy_f_reflect, iters, trials);
  double f_dual = bench_scalar(dheavy_f_dual,    iters, trials);

  double g_hand = bench_grad(grad_heavy_g_hand,        gi, trials);
  double g_refl = bench_grad(grad_heavy_g_reflect,     gi, trials);
  double g_fwd  = bench_grad(grad_heavy_g_reflect_fwd, gi, trials);
  double g_tape = bench_grad(grad_heavy_g_tape,        gi, trials);

  if (csv) {
    std::printf("group,method,ns\n");
    std::printf("forward,hand,%.3f\n", f_hand);
    std::printf("forward,reflection,%.3f\n", f_refl);
    std::printf("forward,dual,%.3f\n", f_dual);
    std::printf("gradient,hand,%.3f\n", g_hand);
    std::printf("gradient,reflection_reverse,%.3f\n", g_refl);
    std::printf("gradient,reflection_forward,%.3f\n", g_fwd);
    std::printf("gradient,runtime_tape,%.3f\n", g_tape);
    return 0;
  }

  std::printf("iters=%zu (gradient=%zu), trials=%d, min ns/call\n\n", iters, gi, trials);
  std::printf("FORWARD  f'(x)  (1 input):\n");
  std::printf("  hand-written        %8.2f ns   %.2fx\n", f_hand, f_hand/f_hand);
  std::printf("  reflection AD       %8.2f ns   %.2fx\n", f_refl, f_refl/f_hand);
  std::printf("  dual numbers (OO)   %8.2f ns   %.2fx\n\n", f_dual, f_dual/f_hand);
  std::printf("GRADIENT  grad g  (4 inputs -> scalar):\n");
  std::printf("  hand-written        %8.2f ns   %.2fx\n", g_hand, g_hand/g_hand);
  std::printf("  reflection reverse  %8.2f ns   %.2fx\n", g_refl, g_refl/g_hand);
  std::printf("  reflection forward  %8.2f ns   %.2fx\n", g_fwd,  g_fwd/g_hand);
  std::printf("  runtime tape (OO)   %8.2f ns   %.2fx\n", g_tape, g_tape/g_hand);
  return 0;
}
