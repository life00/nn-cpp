#pragma once

#include <cstddef>
#include <random>
#include <vector>

enum class Act { sigmoid, relu };

double act(Act sigma, double xi);            // y_j = sigma_j(xi_j)
double dact(Act sigma, double xi, double y); // sigma_j'(xi_j)

// indexed by layer l = 0..L; slot 0 unused except y[0] = x_k
struct Net {
  std::vector<std::size_t> widths;                  // widths[l] = |layer l|
  std::vector<Act> sigma;                           // sigma[l] = sigma_j
  std::vector<std::vector<double>> xi;              // xi[l][j] = xi_j
  std::vector<std::vector<double>> y;               // y[l][j] = y_j
  std::vector<std::vector<double>> dey;             // dey[l][j] = dE_k/dy_j
  std::vector<std::vector<std::vector<double>>> w;  // w[l][j][i] = w_ji
  std::vector<std::vector<double>> b;               // b[l][j] = w_j0
  std::vector<std::vector<std::vector<double>>> dew; // dew[l][j][i] = dE/dw_ji
  std::vector<std::vector<double>> deb;             // deb[l][j] = dE/dw_j0
};

// w[l]: widths[l] x widths[l-1]; w_ji, w_j0 ~ N(0, 1/widths[l-1])
bool setup(Net &net, const std::vector<std::size_t> &widths,
           const std::vector<Act> &acts, std::mt19937 &rng);

// dE/dw_ji := 0
void clear_grad(Net &net);

// xi_j = sum_{i in j<-} w_ji y_i (y_0 = 1), y_j = sigma_j(xi_j)
void forward(Net &net, const double *x);

// dE/dw_ji += (1/p) dE_k/dw_ji, dE_k/dw_ji = dE_k/dy_j sigma_j'(xi_j) y_i
void backward(Net &net, const double *d, std::size_t p);

// w_ji := w_ji - eps dE/dw_ji
void step(Net &net, double eps);
