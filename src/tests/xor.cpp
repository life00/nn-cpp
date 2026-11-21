#include "../net.hpp"

#include <cstdlib>
#include <iostream>
#include <random>

int main() {
  // T = {(x_k, d_k) | k = 1..p}, p = 4
  const double x[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; // x_k
  const double d[4][1] = {{0}, {1}, {1}, {0}};             // d_k
  const std::size_t p = 4;
  const double eps = 0.5; // eps(t)

  std::mt19937 rng(0); // seed in {0, 1, 2, 3, 42}
  Net net;
  if (!setup(net, {2, 2, 1}, {Act::sigmoid, Act::sigmoid}, rng)) {
    exit(1);
  }

  std::cout << "forward:\n";
  double E_before = 0.0;
  for (std::size_t k = 0; k < p; ++k) {
    forward(net, x[k]);
    const double y = net.y.back()[0];
    E_before += 0.5 * (y - d[k][0]) * (y - d[k][0]) / p; // E = (1/p) sum_k E_k
    std::cout << "x_k = (" << x[k][0] << ", " << x[k][1]
              << ")  d_k = " << d[k][0] << "  y = " << y << '\n';
  }

  clear_grad(net); // dE/dw_ji := 0
  for (std::size_t k = 0; k < p; ++k) {
    forward(net, x[k]);
    backward(net, d[k], p); // dE/dw_ji += (1/p) dE_k/dw_ji
  }
  std::cout << "backward:\n";
  for (std::size_t l = 1; l < net.widths.size(); ++l) {
    for (std::size_t j = 0; j < net.widths[l]; ++j) {
      std::cout << "l=" << l << " j=" << j << "  w_j0 = " << net.b[l][j]
                << "  w_ji =";
      for (const double w_ji : net.w[l][j]) {
        std::cout << ' ' << w_ji;
      }
      std::cout << "  dE/dw_j0 = " << net.deb[l][j] << "  dE/dw_ji =";
      for (const double dew_ji : net.dew[l][j]) {
        std::cout << ' ' << dew_ji;
      }
      std::cout << '\n';
    }
  }

  step(net, eps); // w_ji := w_ji - eps dE/dw_ji
  std::cout << "step:\n";
  for (std::size_t l = 1; l < net.widths.size(); ++l) {
    for (std::size_t j = 0; j < net.widths[l]; ++j) {
      std::cout << "l=" << l << " j=" << j << "  w_j0 = " << net.b[l][j]
                << "  w_ji =";
      for (const double w_ji : net.w[l][j]) {
        std::cout << ' ' << w_ji;
      }
      std::cout << '\n';
    }
  }

  std::cout << "forward:\n";
  double E_after = 0.0;
  for (std::size_t k = 0; k < p; ++k) {
    forward(net, x[k]);
    const double y = net.y.back()[0];
    E_after += 0.5 * (y - d[k][0]) * (y - d[k][0]) / p; // E = (1/p) sum_k E_k
    std::cout << "x_k = (" << x[k][0] << ", " << x[k][1]
              << ")  d_k = " << d[k][0] << "  y = " << y << '\n';
  }
  std::cout << "E before = " << E_before << "  E after = " << E_after << '\n';
  return 0;
}
