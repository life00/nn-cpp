#include "io.hpp"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

bool Dataset::fail(const std::string &msg) {
  std::cerr << msg << '\n';
  return false;
}

bool Dataset::load_features(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    return fail(path + ": cannot open");

  pixels.clear();
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
        return fail(path + ":" + std::to_string(pixels.size() + 1) +
                    ": bad value " + std::to_string(v));
      row_px.push_back(uint8_t(v));
      if (*end != ',')
        break; // last field of the row
      p = end + 1;
    }
    const size_t width = pixels.empty() ? 0 : pixels.front().size();
    if (row_px.size() < 2 || (width != 0 && row_px.size() != width))
      return fail(path + ":" + std::to_string(pixels.size() + 1) + ": " +
                  std::to_string(row_px.size()) + " fields");
    pixels.push_back(row_px);
  }
  if (pixels.empty())
    return fail(path + ": no data");
  return true;
}

bool Dataset::load_targets(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    return fail(path + ": cannot open");

  labels.clear();
  std::string line;
  while (std::getline(in, line)) {
    char *end;
    const long v = strtol(line.c_str(), &end, 10);
    while (*end == '\r')
      ++end;
    if (end == line.c_str() || *end != '\0' || v < 0 || v > 9)
      return fail(path + ":" + std::to_string(labels.size() + 1) +
                  ": bad label");
    labels.push_back(uint8_t(v));
  }
  if (pixels.empty())
    return fail(path + ": load features first");
  if (labels.size() != pixels.size())
    return fail(path + ": " + std::to_string(labels.size()) +
                " targets for " + std::to_string(pixels.size()) + " rows");
  return true;
}

Dataset Dataset::create_validation(const double share) {
  Dataset val;
  const size_t n = pixels.size();
  // val is the tail of this, so this || val is the original input order
  const size_t n_val = size_t(share * double(n));
  if (n_val == 0 || n_val >= n || labels.size() != n)
    return val; // no split
  const size_t n_train = n - n_val;
  val.pixels.assign(std::make_move_iterator(pixels.begin() + n_train),
                    std::make_move_iterator(pixels.end()));
  val.labels.assign(std::make_move_iterator(labels.begin() + n_train),
                    std::make_move_iterator(labels.end()));
  pixels.resize(n_train);
  labels.resize(n_train);
  assert(val.pixels.size() == n_val && pixels.size() == n_train);
  assert(val.labels.size() == n_val && labels.size() == n_train);
  return val;
}

const std::vector<std::vector<uint8_t>> &Dataset::features() {
  return pixels;
}

const std::vector<uint8_t> &Dataset::targets() {
  return labels;
}

bool Dataset::save_targets(const std::string &path,
                           const std::vector<uint8_t> &labels) {
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
