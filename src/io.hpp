#pragma once

#include <cstddef>
#include <string>
#include <vector>

class Dataset {
public:
  bool load_features(const std::string &path);
  bool load_targets(const std::string &path);
  // moves the last share of rows into the returned Dataset
  Dataset create_validation(float share);

  const std::vector<std::vector<float>> &features() const;
  const std::vector<std::size_t> &targets() const;
  // share of predictions equal to the labels
  float accuracy(const std::vector<std::size_t> &predictions) const;

  static bool save_predictions(const std::string &path,
                               const std::vector<std::size_t> &predictions);

private:
  static bool fail(const std::string &msg);

  std::vector<std::vector<float>>
      pixels; // one image per row, x_k = pixel / 255, all rows the same length
  std::vector<std::size_t> labels; // one class 0-9 per row
};
