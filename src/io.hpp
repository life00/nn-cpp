#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Dataset {
public:
  bool load_features(const std::string &path);
  bool load_targets(const std::string &path);
  // moves the last share of rows into the returned Dataset
  Dataset create_validation(double share);

  const std::vector<std::vector<uint8_t>> &features();
  const std::vector<uint8_t> &targets();

  static bool save_targets(const std::string &path,
                           const std::vector<uint8_t> &labels);

private:
  static bool fail(const std::string &msg);

  std::vector<std::vector<uint8_t>>
      pixels;                  // one image per row, all rows the same length
  std::vector<uint8_t> labels; // one per row
};
