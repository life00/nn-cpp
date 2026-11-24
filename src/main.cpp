#include "io.hpp"
#include "net.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Config {
  std::string train_pixels;
  std::string train_labels;
  std::string test_pixels;
  std::string train_predictions;
  std::string test_predictions;
  float val_share = 0;
  std::size_t seed = 0;
  std::size_t epochs = 0;
  std::size_t batch_size = 0;
  float epsilon = 0;
  std::vector<std::size_t> widths;
  std::vector<Act> acts;
};

int main() {
  Config cfg = Config{
      // data parameters
      .train_pixels = "data/fashion_mnist_train_vectors.csv",
      .train_labels = "data/fashion_mnist_train_labels.csv",
      .test_pixels = "data/fashion_mnist_test_vectors.csv",
      .train_predictions = "train_predictions.csv",
      .test_predictions = "test_predictions.csv",
      .val_share = 0.0f,
      // training parameters
      .seed = 42,
      .epochs = 50,
      .batch_size = 32,
      .epsilon = 0.5f,
      .widths = {784, 128, 10},
      .acts = {Act::relu, Act::sigmoid},
  };

  std::mt19937 rng(cfg.seed);

  // load data
  Dataset train;
  Dataset test;
  if (!train.load_features(cfg.train_pixels)) {
    exit(1);
  }
  if (!train.load_targets(cfg.train_labels)) {
    exit(1);
  }
  if (!test.load_features(cfg.test_pixels)) {
    exit(1);
  }
  Dataset val = train.create_validation(cfg.val_share);

  std::cout << "train " << train.features().size() << ", val "
            << val.features().size() << ", test " << test.features().size()
            << '\n';

  // data validation
  if (train.features().front().size() != cfg.widths.front() ||
      test.features().front().size() != cfg.widths.front() ||
      cfg.widths.back() != 10) {
    std::cerr << "network widths do not match the data\n";
    exit(1);
  }

  Network net(cfg.widths, cfg.acts, rng);

  // order[m] = row of the m-th training step, reshuffled every epoch
  std::vector<std::size_t> order(train.features().size());
  for (std::size_t k = 0; k < order.size(); ++k) {
    order[k] = k;
  }

  // reused every epoch
  std::vector<std::size_t> val_pred;

  // training loop
  for (std::size_t t = 0; t < cfg.epochs; ++t) {
    std::shuffle(order.begin(), order.end(), rng);

    // linear epsilon decay
    // eps(t) = eps(0) (1 - t/epochs)
    const float eps = cfg.epsilon * (1.0f - float(t) / cfg.epochs);

    float E = 0.0f; // E = (1/n) sum_k E_k

    for (std::size_t first = 0; first < order.size(); first += cfg.batch_size) {
      const std::size_t p =
          std::min(cfg.batch_size, order.size() - first); // may be short

      // one minibatch
      // accumulate p gradients, then one step
      net.clear_grad();
      for (std::size_t m = first; m < first + p; ++m) {
        net.forward(train.features()[order[m]]);
        E += net.loss(train.targets()[order[m]]) / order.size();
        net.backward(train.targets()[order[m]], p);
      }
      net.step(eps);
    }

    // training info
    std::cout << "epoch " << t + 1 << "  E = " << E;

    if (!val.features().empty()) {
      val_pred.clear();
      net.predict(val.features(), val_pred);
      std::cout << "  val acc = " << val.accuracy(val_pred);
    }
    std::cout << '\n';
  }

  // predict and save results
  std::vector<std::size_t> train_pred;
  net.predict(train.features(), train_pred);
  net.predict(val.features(),
              train_pred); // val is the tail of train

  std::vector<std::size_t> test_pred;
  net.predict(test.features(), test_pred);

  if (!Dataset::save_predictions(cfg.train_predictions, train_pred) ||
      !Dataset::save_predictions(cfg.test_predictions, test_pred)) {
    exit(1);
  }

  return 0;
}
