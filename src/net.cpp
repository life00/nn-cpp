#include "net.hpp"
#include "io.hpp"

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

bool setup(Net &net, const std::vector<std::size_t> &widths,
           const std::vector<Act> &acts, std::mt19937 &rng) {
  if (widths.size() < 2) {
    return fail("need an input and an output layer");
  }
  if (acts.size() != widths.size() - 1) {
    return fail("need one activation per layer 1..L");
  }
  for (const std::size_t n : widths) {
    if (n == 0) {
      return fail("layer of width 0");
    }
  }

  const std::size_t L = widths.size() - 1;
  net.widths = widths;
  net.sigma.assign(1, Act::sigmoid); // sigma[0] unused
  net.sigma.insert(net.sigma.end(), acts.begin(), acts.end());
  net.xi.assign(L + 1, {});
  net.y.assign(L + 1, {});
  net.ey.assign(L + 1, {});
  net.w.assign(L + 1, {});
  net.b.assign(L + 1, {});
  net.gw.assign(L + 1, {});
  net.gb.assign(L + 1, {});
  net.y[0].assign(widths[0], 0.0); // y[0] = x_k

  for (std::size_t l = 1; l <= L; ++l) {
    const std::size_t n = widths[l], n_in = widths[l - 1];
    net.xi[l].assign(n, 0.0);
    net.y[l].assign(n, 0.0);
    net.ey[l].assign(n, 0.0);
    net.b[l].assign(n, 0.0);
    net.gb[l].assign(n, 0.0);
    net.w[l].assign(n, std::vector<double>(n_in, 0.0));
    net.gw[l].assign(n, std::vector<double>(n_in, 0.0));

    // Normal LeCun initialization
    std::normal_distribution<double> N(0.0,
                                       std::sqrt(1.0 / n_in)); // N(0, 1/n_in)
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < n_in; ++i) {
        net.w[l][j][i] = N(rng); // w_ji
      }
      net.b[l][j] = N(rng); // w_j0
    }
  }
  return true;
}

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
