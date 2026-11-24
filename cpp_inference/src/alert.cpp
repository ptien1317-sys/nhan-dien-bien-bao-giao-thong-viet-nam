#include "alert.hpp"

#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace its {

AlertManager::AlertManager(float conf_threshold, int cooldown_ms, int banner_ms,
                           const std::vector<std::string>& labels)
    : conf_thr_(conf_threshold), cooldown_ms_(cooldown_ms), banner_ms_(banner_ms), labels_(labels) {}

std::string AlertManager::nameOf(int id) const {
    if (id >= 0 && id < static_cast<int>(labels_.size())) return labels_[id];
    return "class_" + std::to_string(id);
}

std::vector<Alert> AlertManager::evaluate(const std::vector<Detection>& detections) {
    std::vector<Alert> alerts;
    const auto now = Clock::now();
    for (const auto& d : detections) {
        if (d.confidence < conf_thr_) continue;              // loc theo nguong tin cay
        auto it = last_alert_.find(d.class_id);
        if (it != last_alert_.end()) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second).count();
            if (elapsed < cooldown_ms_) continue;            // cung mot bien bao -> khong lap lai canh bao
        }
        last_alert_[d.class_id] = now;
        Alert a{d.class_id, nameOf(d.class_id), d.confidence, ""};
        a.message = "CANH BAO: " + a.label;
        alerts.push_back(a);
        banner_text_ = a.message;
        banner_until_ = now + std::chrono::milliseconds(banner_ms_);
    }
    return alerts;
}

void AlertManager::draw(cv::Mat& frame, const std::vector<Detection>& detections, double fps) {
    for (const auto& d : detections) {
        const bool strong = d.confidence >= conf_thr_;
        const cv::Scalar color = strong ? cv::Scalar(0, 200, 0) : cv::Scalar(160, 160, 160);
        cv::rectangle(frame, d.box, color, 2);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s %.2f", nameOf(d.class_id).c_str(), d.confidence);
        int base = 0;
        const cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        const int ty = std::max(d.box.y, ts.height + 4);
        cv::rectangle(frame, cv::Rect(d.box.x, ty - ts.height - 4, ts.width + 4, ts.height + 6), color, cv::FILLED);
        cv::putText(frame, buf, cv::Point(d.box.x + 2, ty - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    if (Clock::now() < banner_until_) {                       // banner canh bao phia tren
        cv::rectangle(frame, cv::Rect(0, 0, frame.cols, 44), cv::Scalar(0, 0, 200), cv::FILLED);
        cv::putText(frame, banner_text_, cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    char fbuf[32];
    std::snprintf(fbuf, sizeof(fbuf), "FPS: %.1f", fps);
    cv::putText(frame, fbuf, cv::Point(10, frame.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
}

}  // namespace its
