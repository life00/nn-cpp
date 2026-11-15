#pragma once

#include <cstddef>
#include <random>
#include <vector>

// xi_j = inner potential, y_j = output of node j,
// w_ji = weight i -> j, w_j0 = b_j (bias),
// sigma_j = activation of node j.

enum class Act { sigmoid, relu };

double act(Act sigma, double xi);  // y_j = sigma_j(xi_j)
double dact(Act sigma, double xi); // sigma_j'(xi_j)

struct Net {
  std::vector<std::size_t> widths;                  // widths[0] = input size
  std::vector<Act> acts;                            // sigma per non-input layer
  std::vector<std::vector<std::vector<double>>> w;  // w[l][j][i] = w_ji
  std::vector<std::vector<std::vector<double>>> gw; // dE/dw_ji
  std::vector<std::vector<double>> b;               // b[l][j] = w_j0
  std::vector<std::vector<double>> gb;              // dE/dw_j0
  std::vector<std::vector<std::vector<double>>> xi; // [l][k][j] = xi_j of x_k
  std::vector<std::vector<std::vector<double>>> y;  // [l][k][j] = y_j; y[0] = x
  std::size_t batch = 0; // scratch capacity; calls pass n <= batch
};

// sizes every buffer once, asserts the shapes, initializes w/b from rng
void setup(Net &net, const std::vector<std::size_t> &widths,
           const std::vector<Act> &acts, std::size_t batch, std::mt19937 &rng);

// fills xi/y for the first n rows of x; the output lands in y.back()
void forward(Net &net, const double *x, std::size_t n);

// fills gw/gb = (1/n) * sum_k dE_k/dw from xi/y and the d_kj in d
void backward(Net &net, const unsigned char *d, std::size_t n);

// w_ji -= lr * gw, b_j -= lr * gb
void step(Net &net, double lr);
