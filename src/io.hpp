#pragma once

#include <string>
#include <vector>

class Dataset {
public:
  bool load_features(const std::string &path);
  bool load_targets(const std::string &path);
  // moves the last share of rows into the returned Dataset
  Dataset create_validation(double share);

  const std::vector<std::vector<double>> &features() const;
  const std::vector<double> &targets() const;
  // share of predictions equal to the labels
  double accuracy(const std::vector<double> &predictions) const;

  static bool save_predictions(const std::string &path,
                               const std::vector<double> &predictions);

private:
  static bool fail(const std::string &msg);

  std::vector<std::vector<double>>
      pixels; // one image per row, x_k = pixel / 255, all rows the same length
  std::vector<double> labels; // one class 0-9 per row
};
