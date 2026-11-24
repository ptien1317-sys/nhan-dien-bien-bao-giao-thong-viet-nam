#pragma once
// detector.hpp - Dinh nghia lop Detector (phat hien + phan loai bien bao bang best.onnx)
//
// Gia dinh dau ra mo hinh theo dinh dang Ultralytics YOLOv8/YOLO11 xuat ONNX:
//   [1, 4 + nc, N]  (hoac [1, N, 4 + nc]) ; 4 so dau = (cx, cy, w, h) tinh theo pixel
//   cua anh dau vao (letterbox), nc so tiep theo = diem so tung lop (khong co objectness).
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "its_config.hpp"

namespace its {

struct Detection {
    int class_id = -1;
    float confidence = 0.f;
    cv::Rect box;            // toa do tren khung hinh goc
};

// Thoi gian tung giai doan cua mot khung hinh (ms)
struct StageTimings {
    double preprocess_ms  = 0;
    double inference_ms   = 0;
    double postprocess_ms = 0;
    double total() const { return preprocess_ms + inference_ms + postprocess_ms; }
};

class Detector {
public:
    explicit Detector(const RuntimeConfig& cfg);

    // Phat hien bien bao tren 1 khung hinh BGR. Neu timings != nullptr thi ghi lai thoi gian.
    std::vector<Detection> detect(const cv::Mat& frame, StageTimings* timings = nullptr);

    // Chay n lan tren anh trong de "lam nong" mo hinh (cap phat bo nho, nap kernel).
    void warmup(int n);

    int numClasses() const { return num_classes_; }
    const std::string& className(int id) const;
    const std::vector<std::string>& labels() const { return labels_; }

private:
    cv::Mat preprocess(const cv::Mat& frame, float& scale, int& pad_x, int& pad_y);
    std::vector<Detection> postprocess(const cv::Mat& output, const cv::Size& frame_size,
                                       float scale, int pad_x, int pad_y);

    RuntimeConfig cfg_;
    cv::dnn::Net net_;
    std::vector<std::string> labels_;
    std::vector<std::string> out_names_;
    cv::Mat canvas_;          // bo dem letterbox tai su dung giua cac khung hinh
    int num_classes_ = 0;
};

}  // namespace its
