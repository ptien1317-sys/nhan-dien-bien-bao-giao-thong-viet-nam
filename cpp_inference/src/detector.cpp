#include "detector.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "its_utils.hpp"

namespace its {

Detector::Detector(const RuntimeConfig& cfg) : cfg_(cfg) {
    net_ = cv::dnn::readNetFromONNX(cfg_.model_path);
    if (net_.empty())
        throw std::runtime_error("Khong nap duoc mo hinh ONNX: " + cfg_.model_path);

    if (cfg_.target == "cuda" || cfg_.target == "cuda_fp16") {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cfg_.target == "cuda" ? cv::dnn::DNN_TARGET_CUDA
                                                       : cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    if (cfg_.num_threads > 0) cv::setNumThreads(cfg_.num_threads);

    out_names_ = net_.getUnconnectedOutLayersNames();
    labels_ = loadLabels(cfg_.labels_path);
    if (labels_.empty())
        std::cerr << "[Detector] Canh bao: khong doc duoc nhan tu " << cfg_.labels_path
                  << " - se hien thi 'class_<id>'.\n";
    canvas_ = cv::Mat(cfg_.input_height, cfg_.input_width, CV_8UC3);
}

const std::string& Detector::className(int id) const {
    static const std::string unknown = "unknown";
    if (id >= 0 && id < static_cast<int>(labels_.size())) return labels_[id];
    return unknown;
}

// Letterbox: giu ti le khung hinh, de phan con thua bang mau xam 114 (giong luc huan luyen YOLO).
cv::Mat Detector::preprocess(const cv::Mat& frame, float& scale, int& pad_x, int& pad_y) {
    scale = std::min(cfg_.input_width / static_cast<float>(frame.cols),
                     cfg_.input_height / static_cast<float>(frame.rows));
    const int nw = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
    const int nh = std::max(1, static_cast<int>(std::round(frame.rows * scale)));
    pad_x = (cfg_.input_width - nw) / 2;
    pad_y = (cfg_.input_height - nh) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    canvas_.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(canvas_(cv::Rect(pad_x, pad_y, nw, nh)));

    // BGR -> RGB, chia 255, doi sang NCHW
    return cv::dnn::blobFromImage(canvas_, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);
}

std::vector<Detection> Detector::postprocess(const cv::Mat& out, const cv::Size& frame_size,
                                             float scale, int pad_x, int pad_y) {
    CV_Assert(out.dims == 3 && out.type() == CV_32F);
    const int d1 = out.size[1], d2 = out.size[2];
    cv::Mat raw(d1, d2, CV_32F, const_cast<float*>(out.ptr<float>()));
    // Chuan hoa ve dang N x (4 + nc)
    cv::Mat pred = (d1 < d2) ? cv::Mat(raw.t()) : raw;
    const int nc = pred.cols - 4;
    if (nc <= 0) throw std::runtime_error("Dinh dang dau ra ONNX khong hop le");
    num_classes_ = nc;

    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> shifted;   // hop bao dich theo lop de NMS tinh rieng tung lop
    std::vector<float> scores;
    std::vector<int> class_ids;
    const cv::Rect frame_rect(0, 0, frame_size.width, frame_size.height);

    for (int i = 0; i < pred.rows; ++i) {
        const float* p = pred.ptr<float>(i);
        int best = 0;
        float best_score = p[4];
        for (int c = 1; c < nc; ++c) {
            if (p[4 + c] > best_score) { best_score = p[4 + c]; best = c; }
        }
        if (best_score < cfg_.conf_threshold) continue;

        const float x1 = (p[0] - p[2] / 2.f - pad_x) / scale;
        const float y1 = (p[1] - p[3] / 2.f - pad_y) / scale;
        cv::Rect box(cvRound(x1), cvRound(y1), cvRound(p[2] / scale), cvRound(p[3] / scale));
        box &= frame_rect;
        if (box.width <= 0 || box.height <= 0) continue;

        boxes.push_back(box);
        shifted.push_back(cv::Rect(box.x + best * 8192, box.y, box.width, box.height));
        scores.push_back(best_score);
        class_ids.push_back(best);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(shifted, scores, cfg_.conf_threshold, cfg_.nms_threshold, keep,
                      1.f, cfg_.max_detections);

    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) result.push_back({class_ids[idx], scores[idx], boxes[idx]});
    return result;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame, StageTimings* timings) {
    Timer t;
    float scale; int pad_x, pad_y;
    cv::Mat blob = preprocess(frame, scale, pad_x, pad_y);
    const double t_pre = t.elapsedMs();

    t.reset();
    net_.setInput(blob);
    std::vector<cv::Mat> outs;
    net_.forward(outs, out_names_);
    const double t_inf = t.elapsedMs();

    t.reset();
    std::vector<Detection> dets = postprocess(outs.at(0), frame.size(), scale, pad_x, pad_y);
    const double t_post = t.elapsedMs();

    if (timings) *timings = {t_pre, t_inf, t_post};
    return dets;
}

void Detector::warmup(int n) {
    cv::Mat dummy(cfg_.input_height, cfg_.input_width, CV_8UC3, cv::Scalar(100, 100, 100));
    for (int i = 0; i < n; ++i) detect(dummy);
}

}  // namespace its
