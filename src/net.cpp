#include "net.hpp"

#include <cmath>

double act(const Act sigma, const double xi) {
  switch (sigma) {
  case Act::sigmoid:
    return 1.0 / (1.0 + std::exp(-xi)); // 1 / (1 + e^-xi_j)
  case Act::relu:
    return xi > 0.0 ? xi : 0.0; // max(0, xi_j)
  }
  return 0.0;
}

double dact(const Act sigma, const double xi, const double y) {
  switch (sigma) {
  case Act::sigmoid:
    return y * (1.0 - y); // y_j (1 - y_j)
  case Act::relu:
    return xi > 0.0 ? 1.0 : 0.0; // [xi_j > 0]
  }
  return 0.0;
}

void setup(Net &, const std::vector<std::size_t> &, const std::vector<Act> &,
           std::mt19937 &) {}

// dE/dw_ji := 0
void clear_grad(Net &) {}

// y[0] = x_k; l = 1..L:
//   xi_j = w_j0 + sum_{i in j<-} w_ji y_i
//   y_j = sigma_j(xi_j)
void forward(Net &, const double *) {}

// output:  dE_k/dy_j = y_j - d_kj
// hidden:  dE_k/dy_j = sum_{r in j->} dE_k/dy_r sigma_r'(xi_r) w_rj
// weights: dE/dw_ji += (1/p) dE_k/dy_j sigma_j'(xi_j) y_i   (y_0 = 1)
void backward(Net &, const double *, std::size_t) {}

// w_ji := w_ji - eps dE/dw_ji
void step(Net &, double) {}
