#include "../net.hpp"

#include <cstdlib>
#include <iostream>
#include <random>

int main() {
  // T = {(x_k, d_k) | k = 1..p}, p = 4
  const double x[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; // x_k
  const double d[4][1] = {{0}, {1}, {1}, {0}};             // d_k

  std::mt19937 rng(0); // seed in {0, 1, 2, 3, 42}
  Net net;
  if (!setup(net, {2, 2, 1}, {Act::sigmoid, Act::sigmoid}, rng)) {
    exit(1);
  }

  clear_grad(net); // dE/dw_ji := 0
  for (std::size_t k = 0; k < 4; ++k) {
    forward(net, x[k]);
    std::cout << "x_k = (" << x[k][0] << ", " << x[k][1]
              << ")  d_k = " << d[k][0] << "  y = " << net.y.back()[0] << '\n';
  }
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
  return 0;
}
