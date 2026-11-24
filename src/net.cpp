#include "net.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

Network::Network(const std::vector<std::size_t> &widths,
                 const std::vector<Act> &acts, std::mt19937 &rng)
    : widths(widths) {
  if (!check(widths, acts)) {
    std::exit(1);
  }

  const std::size_t L = widths.size() - 1;
  sigma.assign(1, Act::sigmoid); // sigma[0] unused
  sigma.insert(sigma.end(), acts.begin(), acts.end());
  xi.assign(L + 1, {});
  y.assign(L + 1, {});
  dey.assign(L + 1, {});
  w.assign(L + 1, {});
  b.assign(L + 1, {});
  dew.assign(L + 1, {});
  deb.assign(L + 1, {});
  y[0].assign(widths[0], 0.0); // y[0] = x_k

  for (std::size_t l = 1; l <= L; ++l) {
    const std::size_t n = widths[l], n_in = widths[l - 1];
    xi[l].assign(n, 0.0);
    y[l].assign(n, 0.0);
    dey[l].assign(n, 0.0);
    b[l].assign(n, 0.0);
    deb[l].assign(n, 0.0);
    w[l].assign(n, std::vector<double>(n_in, 0.0));
    dew[l].assign(n, std::vector<double>(n_in, 0.0));

    // Normal LeCun initialization
    std::normal_distribution<double> N(0.0,
                                       std::sqrt(1.0 / n_in)); // N(0, 1/n_in)
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < n_in; ++i) {
        w[l][j][i] = N(rng); // w_ji
      }
      b[l][j] = N(rng); // w_j0
    }
  }
}

void Network::forward(const std::vector<double> &x_k) {
  for (std::size_t i = 0; i < widths[0]; ++i) {
    y[0][i] = x_k[i]; // y[0] = x_k
  }
  for (std::size_t l = 1; l < widths.size(); ++l) {
    for (std::size_t j = 0; j < widths[l]; ++j) {
      double xi_j = b[l][j]; // w_j0 y_0, y_0 = 1
      for (std::size_t i = 0; i < widths[l - 1]; ++i) {
        xi_j += w[l][j][i] * y[l - 1][i]; // + w_ji y_i
      }
      xi[l][j] = xi_j;               // xi_j
      y[l][j] = act(sigma[l], xi_j); // y_j = sigma_j(xi_j)
    }
  }
}

void Network::clear_grad() {
  for (std::size_t l = 1; l < widths.size(); ++l) {
    for (std::vector<double> &dew_j : dew[l]) {
      std::fill(dew_j.begin(), dew_j.end(), 0.0); // dE/dw_ji := 0
    }
    std::fill(deb[l].begin(), deb[l].end(), 0.0); // dE/dw_j0 := 0
  }
}

void Network::backward(const double d_k, const std::size_t p) {
  const std::size_t L = widths.size() - 1;

  for (std::size_t j = 0; j < widths[L]; ++j) {
    const double d_kj = j == d_k ? 1.0 : 0.0;
    dey[L][j] = y[L][j] - d_kj; // dE_k/dy_j = y_j - d_kj
  }

  for (std::size_t l = L - 1; l >= 1; --l) {
    for (std::size_t j = 0; j < widths[l]; ++j) {
      double dey_j = 0.0;
      for (std::size_t r = 0; r < widths[l + 1]; ++r) {
        const double dyxi = dact(sigma[l + 1], xi[l + 1][r],
                                 y[l + 1][r]); // dy_r/dxi_r = sigma_r'(xi_r)
        dey_j += dey[l + 1][r] * dyxi *
                 w[l + 1][r][j]; // dE_k/dy_r sigma_r'(xi_r) w_rj
      }
      dey[l][j] = dey_j; // dE_k/dy_j = sum_{r in j->} ...
    }
  }

  for (std::size_t l = 1; l <= L; ++l) {
    for (std::size_t j = 0; j < widths[l]; ++j) {
      const double dyxi =
          dact(sigma[l], xi[l][j], y[l][j]); // dy_j/dxi_j = sigma_j'(xi_j)
      for (std::size_t i = 0; i < widths[l - 1]; ++i) {
        const double dew_k =
            dey[l][j] * dyxi *
            y[l - 1][i];           // dE_k/dw_ji = dE_k/dy_j sigma_j'(xi_j) y_i
        dew[l][j][i] += dew_k / p; // dE/dw_ji += (1/p) dE_k/dw_ji
      }
      const double deb_k =
          dey[l][j] * dyxi *
          1.0; // dE_k/dw_j0 = dE_k/dy_j sigma_j'(xi_j) y_0, y_0 = 1
      deb[l][j] += deb_k / p; // dE/dw_j0 += (1/p) dE_k/dw_j0
    }
  }
}

void Network::step(const double eps) {
  for (std::size_t l = 1; l < widths.size(); ++l) {
    for (std::size_t j = 0; j < widths[l]; ++j) {
      for (std::size_t i = 0; i < widths[l - 1]; ++i) {
        w[l][j][i] -= eps * dew[l][j][i]; // w_ji := w_ji - eps dE/dw_ji
      }
      b[l][j] -= eps * deb[l][j]; // w_j0 := w_j0 - eps dE/dw_j0
    }
  }
}

double Network::loss(const double d_k) {
  const std::size_t L = widths.size() - 1;
  double E_k = 0.0; // E_k = 1/2 sum_j (y_j - d_kj)^2
  for (std::size_t j = 0; j < widths[L]; ++j) {
    const double d_kj = j == d_k ? 1.0 : 0.0;
    E_k += 0.5 * (y[L][j] - d_kj) * (y[L][j] - d_kj);
  }
  return E_k;
}

void Network::predict(const std::vector<std::vector<double>> &x,
                      std::vector<double> &out) {
  const std::size_t L = widths.size() - 1;
  for (const std::vector<double> &x_k : x) {
    forward(x_k);
    out.push_back(double(std::max_element(y[L].begin(), y[L].end()) -
                         y[L].begin())); // argmax_j y_j
  }
}

bool Network::fail(const std::string &msg) {
  std::cerr << msg << '\n';
  return false;
}

bool Network::check(const std::vector<std::size_t> &widths,
                    const std::vector<Act> &acts) {
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
  return true;
}

double Network::act(const Act sigma, const double xi) {
  switch (sigma) {
  case Act::sigmoid:
    return 1.0 / (1.0 + std::exp(-xi)); // 1 / (1 + e^-xi_j)
  case Act::relu:
    return xi > 0.0 ? xi : 0.0; // max(0, xi_j)
  }
  return 0.0;
}

double Network::dact(const Act sigma, const double xi, const double y) {
  switch (sigma) {
  case Act::sigmoid:
    return y * (1.0 - y); // y_j (1 - y_j)
  case Act::relu:
    return xi > 0.0 ? 1.0 : 0.0; // [xi_j > 0]
  }
  return 0.0;
}
