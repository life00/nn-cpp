#include "net.hpp"
#include "io.hpp"

#include <algorithm>
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
  net.dey.assign(L + 1, {});
  net.w.assign(L + 1, {});
  net.b.assign(L + 1, {});
  net.dew.assign(L + 1, {});
  net.deb.assign(L + 1, {});
  net.y[0].assign(widths[0], 0.0); // y[0] = x_k

  for (std::size_t l = 1; l <= L; ++l) {
    const std::size_t n = widths[l], n_in = widths[l - 1];
    net.xi[l].assign(n, 0.0);
    net.y[l].assign(n, 0.0);
    net.dey[l].assign(n, 0.0);
    net.b[l].assign(n, 0.0);
    net.deb[l].assign(n, 0.0);
    net.w[l].assign(n, std::vector<double>(n_in, 0.0));
    net.dew[l].assign(n, std::vector<double>(n_in, 0.0));

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

void forward(Net &net, const double *x) {
  for (std::size_t i = 0; i < net.widths[0]; ++i) {
    net.y[0][i] = x[i]; // y[0] = x_k
  }
  for (std::size_t l = 1; l < net.widths.size(); ++l) {
    for (std::size_t j = 0; j < net.widths[l]; ++j) {
      double xi = net.b[l][j]; // w_j0 y_0, y_0 = 1
      for (std::size_t i = 0; i < net.widths[l - 1]; ++i) {
        xi += net.w[l][j][i] * net.y[l - 1][i]; // + w_ji y_i
      }
      net.xi[l][j] = xi;                   // xi_j
      net.y[l][j] = act(net.sigma[l], xi); // y_j = sigma_j(xi_j)
    }
  }
}

void clear_grad(Net &net) {
  for (std::size_t l = 1; l < net.widths.size(); ++l) {
    for (std::vector<double> &dew_j : net.dew[l]) {
      std::fill(dew_j.begin(), dew_j.end(), 0.0); // dE/dw_ji := 0
    }
    std::fill(net.deb[l].begin(), net.deb[l].end(), 0.0); // dE/dw_j0 := 0
  }
}

void backward(Net &net, const double *d, const std::size_t p) {
  const std::size_t L = net.widths.size() - 1;

  for (std::size_t j = 0; j < net.widths[L]; ++j) {
    net.dey[L][j] = net.y[L][j] - d[j]; // dE_k/dy_j = y_j - d_kj
  }

  for (std::size_t l = L - 1; l >= 1; --l) {
    for (std::size_t j = 0; j < net.widths[l]; ++j) {
      double dey_j = 0.0;
      for (std::size_t r = 0; r < net.widths[l + 1]; ++r) {
        const double dyxi = dact(net.sigma[l + 1], net.xi[l + 1][r],
                                 net.y[l + 1][r]); // dy_r/dxi_r = sigma_r'(xi_r)
        dey_j += net.dey[l + 1][r] * dyxi * net.w[l + 1][r][j]; // dE_k/dy_r sigma_r'(xi_r) w_rj
      }
      net.dey[l][j] = dey_j; // dE_k/dy_j = sum_{r in j->} ...
    }
  }

  for (std::size_t l = 1; l <= L; ++l) {
    for (std::size_t j = 0; j < net.widths[l]; ++j) {
      const double dyxi = dact(net.sigma[l], net.xi[l][j], net.y[l][j]); // dy_j/dxi_j = sigma_j'(xi_j)
      for (std::size_t i = 0; i < net.widths[l - 1]; ++i) {
        const double dew_k = net.dey[l][j] * dyxi * net.y[l - 1][i]; // dE_k/dw_ji = dE_k/dy_j sigma_j'(xi_j) y_i
        net.dew[l][j][i] += dew_k / p; // dE/dw_ji += (1/p) dE_k/dw_ji
      }
      const double deb_k = net.dey[l][j] * dyxi * 1.0; // dE_k/dw_j0 = dE_k/dy_j sigma_j'(xi_j) y_0, y_0 = 1
      net.deb[l][j] += deb_k / p; // dE/dw_j0 += (1/p) dE_k/dw_j0
    }
  }
}

void step(Net &net, const double eps) {
  for (std::size_t l = 1; l < net.widths.size(); ++l) {
    for (std::size_t j = 0; j < net.widths[l]; ++j) {
      for (std::size_t i = 0; i < net.widths[l - 1]; ++i) {
        net.w[l][j][i] -= eps * net.dew[l][j][i]; // w_ji := w_ji - eps dE/dw_ji
      }
      net.b[l][j] -= eps * net.deb[l][j]; // w_j0 := w_j0 - eps dE/dw_j0
    }
  }
}
