#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Config {
  int seed = 0;
  int epochs = 0;
  std::string train_pixels;
  std::string train_labels;
  std::string test_pixels;
};

struct Dataset {
  // one image per row, all rows the same length
  std::vector<std::vector<uint8_t>> features;
  std::vector<uint8_t> targets; // one per row
};

static bool fail(const std::string &msg) {
  std::cerr << msg << '\n';
  return false;
}

bool load_pixels(const std::string &path, Dataset &ds) {
  std::ifstream in(path);
  if (!in)
    return fail(path + ": cannot open");

  ds.features.clear();
  std::vector<uint8_t> row_px;
  row_px.reserve(1024);
  std::string line;
  while (std::getline(in, line)) {
    row_px.clear();
    const char *p = line.c_str();
    for (;;) {
      char *end;
      const long v = strtol(p, &end, 10);
      if (end == p)
        break; // nothing left on this line
      if (v < 0 || v > 255)
        return fail(path + ":" + std::to_string(ds.features.size() + 1) +
                    ": bad value " + std::to_string(v));
      row_px.push_back(uint8_t(v));
      if (*end != ',')
        break; // last field of the row
      p = end + 1;
    }
    const size_t width = ds.features.empty() ? 0 : ds.features.front().size();
    if (row_px.size() < 2 || (width != 0 && row_px.size() != width))
      return fail(path + ":" + std::to_string(ds.features.size() + 1) + ": " +
                  std::to_string(row_px.size()) + " fields");
    ds.features.push_back(row_px);
  }
  if (ds.features.empty())
    return fail(path + ": no data");
  return true;
}

bool load_labels(const std::string &path, Dataset &ds) {
  std::ifstream in(path);
  if (!in)
    return fail(path + ": cannot open");

  ds.targets.clear();
  std::string line;
  while (std::getline(in, line)) {
    char *end;
    const long v = strtol(line.c_str(), &end, 10);
    while (*end == '\r')
      ++end;
    if (end == line.c_str() || *end != '\0' || v < 0 || v > 9)
      return fail(path + ":" + std::to_string(ds.targets.size() + 1) +
                  ": bad label");
    ds.targets.push_back(uint8_t(v));
  }
  if (ds.features.empty())
    return fail(path + ": load features first");
  if (ds.targets.size() != ds.features.size())
    return fail(path + ": " + std::to_string(ds.targets.size()) +
                " targets for " + std::to_string(ds.features.size()) + " rows");
  return true;
}

int main() {
  // 0. config: seed, epochs, data paths
  Config cfg = Config{
      .seed = 0, // allowed: 0,1,2,3,42
      .epochs = 10,
      .train_pixels = "data/fashion_mnist_train_vectors.csv",
      .train_labels = "data/fashion_mnist_train_labels.csv",
      .test_pixels = "data/fashion_mnist_test_vectors.csv",
  };

  // 1. load train vectors + targets
  Dataset train;
  Dataset test;
  if (!load_pixels(cfg.train_pixels, train)) {
    exit(1);
  };
  if (!load_labels(cfg.train_labels, train)) {
    exit(1);
  };
  if (!load_pixels(cfg.test_pixels, test)) {
    exit(1);
  };

  std::cerr << "train " << train.features.size() << "x"
            << train.features.front().size() << ", test "
            << test.features.size() << "x" << test.features.front().size()
            << '\n';

  // 2. carve validation slice out of train
  // 3. create network
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

  std::cout << "thinking...";
  return 0;
}
