#include "io.hpp"
#include "net.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

struct Config {
  int seed = 0;
  int epochs = 0;
  double val_share = 0;
  std::vector<std::size_t> widths; // |layer 0|..|layer L|
  std::vector<Act> acts;           // sigma_j for layers 1..L
  std::string train_pixels;
  std::string train_labels;
  std::string test_pixels;
  std::string train_predictions;
  std::string test_predictions;
};

int main() {
  // 0. config: seed, epochs, topology, data paths
  Config cfg = Config{
      .seed = 0, // allowed: 0,1,2,3,42
      .epochs = 10,
      .val_share = 0.1,         // validation share of training dataset
      .widths = {784, 128, 10}, // input, hidden, output;
      .acts = {Act::sigmoid, Act::sigmoid},
      .train_pixels = "data/fashion_mnist_train_vectors.csv",
      .train_labels = "data/fashion_mnist_train_labels.csv",
      .test_pixels = "data/fashion_mnist_test_vectors.csv",
      .train_predictions = "train_predictions.csv",
      .test_predictions = "test_predictions.csv",
  };

  // 1. load train vectors + targets
  Dataset train;
  Dataset test;
  if (!load_features(cfg.train_pixels, train)) {
    exit(1);
  };
  if (!load_targets(cfg.train_labels, train)) {
    exit(1);
  };
  if (!load_features(cfg.test_pixels, test)) {
    exit(1);
  };
  // 2. carve validation slice out of train
  Dataset val = create_validation(cfg.val_share, train);

  std::cout << "train " << train.features.size() << "x"
            << train.features.front().size() << ", val " << val.features.size()
            << "x" << val.features.front().size() << ", test "
            << test.features.size() << "x" << test.features.front().size()
            << '\n';

  // 3. create network
  std::mt19937 rng(cfg.seed);
  Net net;
  if (!setup(net, cfg.widths, cfg.acts, rng)) {
    exit(1);
  }

  // 4. define loss function
  // 5. define optimizer
  // 6. training loop until epoch == n:
  //    shuffle order with the same RNG
  //    for each batch:
  //      reset gradients
  //      forward pass
  //      loss
  //      backpropagate
  //      optimize weights
  //    evaluate on validation
  //    print epoch, val accuracy, seconds
  //    check 600s budget
  // 7. write train predictions
  // 8. write test predictions
  if (!save_targets(cfg.train_predictions, train.targets) ||
      !save_targets(cfg.test_predictions, test.targets)) {
    exit(1);
  }

  std::cout << "thinking...";
  return 0;
}
