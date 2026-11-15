#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Dataset {
  // one image per row, all rows the same length
  std::vector<std::vector<uint8_t>> features;
  std::vector<uint8_t> targets; // one per row
};

bool fail(const std::string &msg);
bool load_features(const std::string &path, Dataset &ds);
bool load_targets(const std::string &path, Dataset &ds);
Dataset create_validation(double share, Dataset &train);
bool save_targets(const std::string &path, const std::vector<uint8_t> &labels);
