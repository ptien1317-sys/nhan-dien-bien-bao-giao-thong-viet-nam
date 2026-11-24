#pragma once
// its_utils.hpp - Ham tien ich dung chung cho nhanh C++ (header-only)
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>

namespace its {

// Dong ho do thoi gian (ms), dung steady_clock nen khong bi nhay khi chinh gio he thong.
class Timer {
public:
    Timer() { reset(); }
    void reset() { t0_ = Clock::now(); }
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    }
private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0_;
};

// Thong ke mo ta cua mot chuoi do (ms)
struct Summary {
    size_t n = 0;
    double mean = 0, stddev = 0, min = 0, max = 0, p50 = 0, p95 = 0, p99 = 0;
};

inline double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()));
    idx = std::min(std::max<size_t>(idx, 1), sorted.size()) - 1;
    return sorted[idx];
}

inline Summary summarize(std::vector<double> v) {
    Summary s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double acc = 0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(acc / v.size());
    s.min = v.front();
    s.max = v.back();
    s.p50 = percentile(v, 50);
    s.p95 = percentile(v, 95);
    s.p99 = percentile(v, 99);
    return s;
}

inline std::string toJson(const Summary& s) {
    std::ostringstream o;
    o << "{\"mean\": " << s.mean << ", \"std\": " << s.stddev << ", \"min\": " << s.min
      << ", \"p50\": " << s.p50 << ", \"p95\": " << s.p95 << ", \"p99\": " << s.p99
      << ", \"max\": " << s.max << "}";
    return o.str();
}

// Doc file nhan: moi dong mot ten lop (bo qua dong trong, cat ky tu \r)
inline std::vector<std::string> loadLabels(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

inline bool isInteger(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

// Mo nguon video: chuoi chi gom chu so -> chi so camera; nguoc lai -> duong dan file/URL.
inline bool openSource(cv::VideoCapture& cap, const std::string& source) {
    return isInteger(source) ? cap.open(std::stoi(source)) : cap.open(source);
}

}  // namespace its
