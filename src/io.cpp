#include "io.hpp"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

bool fail(const std::string &msg) {
  std::cerr << msg << '\n';
  return false;
}

bool load_features(const std::string &path, Dataset &ds) {
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

bool load_targets(const std::string &path, Dataset &ds) {
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

Dataset create_validation(const double share, Dataset &train) {
  Dataset val;
  const size_t n = train.features.size();
  // val is the tail of train, so train || val is the original input order
  const size_t n_val = size_t(share * double(n));
  if (n_val == 0 || n_val >= n || train.targets.size() != n)
    return val; // no split
  const size_t n_train = n - n_val;
  val.features.assign(std::make_move_iterator(train.features.begin() + n_train),
                      std::make_move_iterator(train.features.end()));
  val.targets.assign(std::make_move_iterator(train.targets.begin() + n_train),
                     std::make_move_iterator(train.targets.end()));
  train.features.resize(n_train);
  train.targets.resize(n_train);
  assert(val.features.size() == n_val && train.features.size() == n_train);
  assert(val.targets.size() == n_val && train.targets.size() == n_train);
  return val;
}

bool save_targets(const std::string &path, const std::vector<uint8_t> &labels) {
  std::ofstream out(path);
  if (!out)
    return fail(path + ": cannot open");
  for (const uint8_t label : labels)
    out << unsigned(label) << '\n'; // uint8_t streams as a char, requires cast
  out.close();
  if (!out)
    return fail(path + ": write failed");
  return true;
}
