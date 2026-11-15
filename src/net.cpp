#include "net.hpp"

#include <cmath>

double act(const Act sigma, const double xi) {
  switch (sigma) {
  case Act::sigmoid:
    return 1.0 / (1.0 + std::exp(-xi));
  case Act::relu:
    return xi > 0.0 ? xi : 0.0;
  }
  return 0.0;
}

double dact(const Act sigma, const double xi) {
  switch (sigma) {
  case Act::sigmoid: {
    const double y = act(sigma, xi); // sigma'(xi) = y(1-y)
    return y * (1.0 - y);
  }
  case Act::relu:
    return xi > 0.0 ? 1.0 : 0.0; // subgradient 0 at 0
  }
  return 0.0;
}

// w[l] and gw[l]: widths[l+1] x widths[l]; b[l], gb[l]: widths[l+1];
// y[l]: batch x widths[l]; xi[l]: batch x widths[l+1]
void setup(Net &, const std::vector<std::size_t> &, const std::vector<Act> &,
           std::size_t, std::mt19937 &) {}

// y[0] = x; then xi[l] = w[l]*y[l] + b[l], y[l+1] = sigma(xi[l])
void forward(Net &, const double *, std::size_t) {}

// output delta y - d, then the j_out sum; each gw entry gets
// delta * sigma'(xi) * y_i. 1/n turns the batch sum into dE/dw
void backward(Net &, const unsigned char *, std::size_t) {}

void step(Net &, double) {}
