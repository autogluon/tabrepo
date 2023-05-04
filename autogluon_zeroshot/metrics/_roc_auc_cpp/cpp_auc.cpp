#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <type_traits>


// Code based on https://github.com/diditforlulz273/fastauc , which is under MIT license

// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes
// that the vectors have equal length)
template <typename tuple_type> void zip(
    const bool* a,
    const float* b,
    const size_t len,
    std::vector<tuple_type> &zipped)
    {
        for(size_t i=0; i<len; ++i)
        {
            zipped.push_back(std::make_tuple(a[i], b[i]));
        }
    }

double trapezoid_area(double x1, double x2, double y1, double y2) {
  double dx = x2 - x1;
  double dy = y2 - y1;
  return dx * y1 + dy * dx / 2.0;
}

template <typename tuple_type> double auc_kernel(float* ts, bool* st, size_t len) {
  // sort the data
  // Zip the vectors together
  std::vector<tuple_type> zipped;
  zipped.reserve(len);
  zip<tuple_type>(st, ts, len, zipped);

  // Sort the vector of pairs
  std::sort(std::begin(zipped), std::end(zipped),
    [&](const auto& a, const auto& b)
    {
        return std::get<1>(a) > std::get<1>(b);
    });

  double fps = 0;
  double tps = 0;
  double last_counted_fps = 0;
  double last_counted_tps = 0;
  double auc = 0;
  for (size_t i=0; i < zipped.size(); ++i) {
    tps += std::get<0>(zipped[i]);
    fps += (1 - std::get<0>(zipped[i]));
    if ((i == zipped.size() - 1) || (std::get<1>(zipped[i+1]) != std::get<1>(zipped[i]))) {
        auc += trapezoid_area(last_counted_fps, fps, last_counted_tps, tps);
        last_counted_fps = fps;
        last_counted_tps = tps;
    }
  }
  return auc / (tps * fps);
}

extern "C" {
    double cpp_auc_ext(float* ts, bool* st, size_t len) {
        return auc_kernel<std::tuple<bool, float>>(ts, st, len);
    }
}