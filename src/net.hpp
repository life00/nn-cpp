#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <vector>

enum class Act { sigmoid, relu };

// indexed by layer l = 0..L; slot 0 unused except y[0] = x_k
class Network {
public:
  // w[l]: widths[l] x widths[l-1]; w_ji, w_j0 ~ N(0, 1/widths[l-1]); exits on
  // bad shape
  Network(const std::vector<std::size_t> &widths, const std::vector<Act> &acts,
          std::mt19937 &rng);

  // dE/dw_ji := 0
  void clear_grad();

  // y[0] = x_k
  // xi_j = sum_{i in j<-} w_ji y_i (y_0 = 1), y_j = sigma_j(xi_j)
  void forward(const std::vector<float> &x_k);

  // dE/dw_ji += (1/p) dE_k/dw_ji, dE_k/dw_ji = dE_k/dy_j sigma_j'(xi_j) y_i
  // d_kj = [j == d_k], d_k = desired class of x_k
  void backward(std::size_t d_k, std::size_t p);

  // w_ji := w_ji - eps dE/dw_ji
  void step(float eps);

  // E_k = 1/2 sum_j (y_j - d_kj)^2, d_kj = [j == d_k]
  float loss(std::size_t d_k);

  // appends predicted class = argmax_j y_j for each x_k
  void predict(const std::vector<std::vector<float>> &x,
               std::vector<std::size_t> &out);

private:
  static bool fail(const std::string &msg);
  static bool check(const std::vector<std::size_t> &widths,
                    const std::vector<Act> &acts);
  static float act(Act sigma, float xi);           // y_j = sigma_j(xi_j)
  static float dact(Act sigma, float xi, float y); // sigma_j'(xi_j)

  std::vector<std::size_t> widths;                   // widths[l] = |layer l|
  std::vector<Act> sigma;                            // sigma[l] = sigma_j
  std::vector<std::vector<float>> xi;               // xi[l][j] = xi_j
  std::vector<std::vector<float>> y;                // y[l][j] = y_j
  std::vector<std::vector<float>> dey;              // dey[l][j] = dE_k/dy_j
  std::vector<std::vector<std::vector<float>>> w;   // w[l][j][i] = w_ji
  std::vector<std::vector<float>> b;                // b[l][j] = w_j0
  std::vector<std::vector<std::vector<float>>> dew; // dew[l][j][i] = dE/dw_ji
  std::vector<std::vector<float>> deb;              // deb[l][j] = dE/dw_j0
};
