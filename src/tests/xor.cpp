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
  const std::size_t epochs = 10000;

  std::mt19937 rng(0);
  Network net({2, 4, 1}, {Act::sigmoid, Act::sigmoid}, rng);

  for (std::size_t t = 0; t < epochs; ++t) {
    net.clear_grad(); // dE/dw_ji := 0
    double E = 0.0;
    for (std::size_t k = 0; k < p; ++k) {
      net.forward(x[k]);
      const double y = net.output()[0];
      E += 0.5 * (y - d[k][0]) * (y - d[k][0]) / p; // E = (1/p) sum_k E_k
      net.backward(d[k], p); // dE/dw_ji += (1/p) dE_k/dw_ji
    }
    net.step(eps); // w_ji := w_ji - eps dE/dw_ji
    if (t % 1000 == 0) {
      std::cout << "t = " << t << "  E = " << E << '\n';
    }
  }

  std::size_t correct = 0;
  for (std::size_t k = 0; k < p; ++k) {
    net.forward(x[k]);
    const double y = net.output()[0];
    correct += (y > 0.5) == (d[k][0] > 0.5); // y rounded == d_k
    std::cout << "x_k = (" << x[k][0] << ", " << x[k][1]
              << ")  d_k = " << d[k][0] << "  y = " << y << '\n';
  }
  std::cout << correct << "/" << p << " correct\n";
  return correct == p ? 0 : 1;
}
