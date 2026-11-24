#pragma once
// alert.hpp - Khoi Hau xu ly & Canh bao ITS (khoi 5 trong pipeline)
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "detector.hpp"

namespace its {

struct Alert {
    int class_id;
    std::string label;
    float confidence;
    std::string message;
};

class AlertManager {
public:
    AlertManager(float conf_threshold, int cooldown_ms, int banner_ms,
                 const std::vector<std::string>& labels);

    // Loc ket qua Confidence >= nguong; tra ve cac canh bao MOI (da ap dung cooldown theo lop).
    std::vector<Alert> evaluate(const std::vector<Detection>& detections);

    // Ve khung bao, nhan, banner canh bao va FPS len khung hinh (chi dung ky tu ASCII).
    void draw(cv::Mat& frame, const std::vector<Detection>& detections, double fps);

private:
    using Clock = std::chrono::steady_clock;
    std::string nameOf(int id) const;

    float conf_thr_;
    int cooldown_ms_, banner_ms_;
    std::vector<std::string> labels_;
    std::unordered_map<int, Clock::time_point> last_alert_;
    std::string banner_text_;
    Clock::time_point banner_until_{};
};

}  // namespace its
