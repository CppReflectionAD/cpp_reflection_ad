// tensor.h — a minimal dependency-free dynamic tensor + op vocabulary, plus a
// small MLP loss. Shared by the tensor demo and the tensor benchmark.
//
// All ops are free functions found by ADL — this is the "vocabulary" the AD
// engine calls. Nothing here is AD-specific; swapping in Eigen/libtorch would
// mean providing these same names.

#ifndef REFLECT_DEMO_TENSOR_H
#define REFLECT_DEMO_TENSOR_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace nn {

struct Tensor {                        // row-major 2-D; scalars are 1x1
  int r = 1, c = 1;
  std::vector<double> d;
  Tensor() : d(1, 0.0) {}
  Tensor(double x) : r(1), c(1), d(1, x) {}
  Tensor(int r_, int c_) : r(r_), c(c_), d(std::size_t(r_) * c_, 0.0) {}
  double &at(int i, int j) { return d[std::size_t(i) * c + j]; }
  double at(int i, int j) const { return d[std::size_t(i) * c + j]; }
};

template <class F>
inline Tensor ew(const Tensor &a, const Tensor &b, F f) {   // elementwise + broadcast
  int R = std::max(a.r, b.r), C = std::max(a.c, b.c);
  Tensor o(R, C);
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j)
      o.at(i, j) = f(a.at(a.r == 1 ? 0 : i, a.c == 1 ? 0 : j),
                     b.at(b.r == 1 ? 0 : i, b.c == 1 ? 0 : j));
  return o;
}
inline Tensor add(const Tensor &a, const Tensor &b) { return ew(a, b, [](double x, double y){ return x + y; }); }
inline Tensor sub(const Tensor &a, const Tensor &b) { return ew(a, b, [](double x, double y){ return x - y; }); }
inline Tensor mul(const Tensor &a, const Tensor &b) { return ew(a, b, [](double x, double y){ return x * y; }); }
inline Tensor neg(const Tensor &a) { Tensor o = a; for (double &v : o.d) v = -v; return o; }
inline Tensor inv(const Tensor &a) { Tensor o = a; for (double &v : o.d) v = 1.0 / v; return o; }

inline Tensor matmul(const Tensor &a, const Tensor &b) {
  Tensor o(a.r, b.c);
  for (int i = 0; i < a.r; ++i)
    for (int k = 0; k < a.c; ++k) {
      double aik = a.at(i, k);
      for (int j = 0; j < b.c; ++j) o.at(i, j) += aik * b.at(k, j);
    }
  return o;
}
inline Tensor transpose(const Tensor &a) {
  Tensor o(a.c, a.r);
  for (int i = 0; i < a.r; ++i)
    for (int j = 0; j < a.c; ++j) o.at(j, i) = a.at(i, j);
  return o;
}
inline Tensor sum(const Tensor &a) { double s = 0; for (double v : a.d) s += v; return Tensor(s); }
inline Tensor relu(const Tensor &a)   { Tensor o = a; for (double &v : o.d) v = v > 0 ? v : 0; return o; }
inline Tensor d_relu(const Tensor &a) { Tensor o = a; for (double &v : o.d) v = v > 0 ? 1 : 0; return o; }
inline Tensor zeros_like(const Tensor &a) { return Tensor(a.r, a.c); }
inline Tensor ones_like(const Tensor &a)  { Tensor o(a.r, a.c); for (double &v : o.d) v = 1.0; return o; }
inline Tensor unbroadcast(const Tensor &g, const Tensor &like) {  // reverse of broadcasting
  Tensor o(like.r, like.c);
  for (int i = 0; i < g.r; ++i)
    for (int j = 0; j < g.c; ++j) o.at(i % like.r, j % like.c) += g.at(i, j);
  return o;
}

// The model: one hidden layer + squared output; loss is a scalar (1x1).
// A-normal form + named ops keep the reflected AST simple.
inline Tensor loss(Tensor x, Tensor W1, Tensor b1, Tensor W2, Tensor b2) {
  Tensor z1 = matmul(x, W1);
  Tensor a1 = add(z1, b1);
  Tensor h  = relu(a1);
  Tensor z2 = matmul(h, W2);
  Tensor o  = add(z2, b2);
  Tensor sq = mul(o, o);
  Tensor L  = sum(sq);
  return L;
}

}  // namespace nn

#endif  // REFLECT_DEMO_TENSOR_H
