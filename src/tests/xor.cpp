#include "../net.hpp"

#include <iostream>
#include <random>
#include <vector>

int main() {
  // T = {(x_k, d_k) | k = 1..p}, p = 4; x_k in [0, 1], d_k = class
  const std::vector<std::vector<float>> x = {
      {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}}; // x_k
  const std::vector<std::size_t> d = {0, 1, 1, 0};             // d_k
  const std::size_t p = x.size();                              // full batch, no shuffle
  const float eps = 0.5f;                                      // eps(t)
  const std::size_t epochs = 10000;

  std::mt19937 rng(0);
  Network net({2, 4, 2}, {Act::sigmoid, Act::sigmoid}, rng);

  for (std::size_t t = 0; t < epochs; ++t) {
    net.clear_grad();
    float E = 0.0f;
    for (std::size_t k = 0; k < p; ++k) {
      net.forward(x[k]);
      E += net.loss(d[k]) / p;
      net.backward(d[k], p);
    }
    net.step(eps);
    if (t % 1000 == 0) {
      std::cout << "t = " << t << "  E = " << E << '\n';
    }
  }

  std::vector<std::size_t> y; // argmax_j y_j per x_k
  net.predict(x, y);
  std::size_t correct = 0;
  for (std::size_t k = 0; k < p; ++k) {
    correct += y[k] == d[k];
    std::cout << "x_k = (" << x[k][0] << ", " << x[k][1] << ")  d_k = " << d[k]
              << " y = " << y[k] << '\n';
  }
  std::cout << correct << "/" << p << " correct\n";
  return correct == p ? 0 : 1;
}
