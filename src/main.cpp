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
  size_t image_size = 0;       // in number of pixels
  std::vector<uint8_t> pixels; // size = rows *  image_size
  std::vector<uint8_t> labels; // one per row
};

static bool fail(const std::string &msg) {
  std::cerr << msg << '\n';
  return false;
}

bool load_pixels(const std::string &path, Dataset &ds) {
  std::ifstream in(path);
  if (!in)
    return fail(path + ": cannot open");

  ds.pixels.clear();
  ds.image_size = 0;
  std::string line;
  size_t row = 0;
  while (std::getline(in, line)) {
    const char *p = line.c_str();
    size_t cols = 0;
    for (;;) {
      char *end;
      const long v = strtol(p, &end, 10);
      if (end == p)
        break; // nothing left on this line
      if (v < 0 || v > 255)
        return fail(path + ":" + std::to_string(row + 1) + ": bad value " +
                    std::to_string(v));
      ds.pixels.push_back(uint8_t(v));
      ++cols;
      if (*end != ',')
        break; // last field of the row
      p = end + 1;
    }
    if (ds.image_size == 0)
      ds.image_size = cols;
    else if (cols != ds.image_size)
      return fail(path + ":" + std::to_string(row + 1) + ": " +
                  std::to_string(cols) + " fields");
    ++row;
  }
  if (row == 0 || ds.image_size < 2)
    return fail(path + ": no data");
  return true;
}

bool load_labels(const std::string &path, Dataset &ds) {
  std::ifstream in(path);
  if (!in)
    return fail(path + ": cannot open");

  ds.labels.clear();
  std::string line;
  while (std::getline(in, line)) {
    char *end;
    const long v = strtol(line.c_str(), &end, 10);
    while (*end == '\r')
      ++end;
    if (end == line.c_str() || *end != '\0' || v < 0 || v > 9)
      return fail(path + ":" + std::to_string(ds.labels.size() + 1) +
                  ": bad label");
    ds.labels.push_back(uint8_t(v));
  }
  if (ds.image_size == 0)
    return fail(path + ": load pixels first");
  const size_t rows = ds.pixels.size() / ds.image_size;
  if (ds.labels.size() != rows)
    return fail(path + ": " + std::to_string(ds.labels.size()) +
                " labels for " + std::to_string(rows) + " rows");
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

  // 1. load train vectors + labels
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

  std::cerr << "train " << train.pixels.size() / train.image_size << "x"
            << train.image_size << ", test "
            << test.pixels.size() / test.image_size << "x" << test.image_size
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
