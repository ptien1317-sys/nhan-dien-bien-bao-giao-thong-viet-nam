#pragma once
// its_config.hpp - Doc cau hinh thoi gian chay tu configs/its_runtime.yaml
#include <string>

namespace its {

struct RuntimeConfig {
    // model
    std::string model_path   = "outputs/checkpoints/best.onnx";
    std::string labels_path  = "outputs/checkpoints/labels.txt";
    int input_width  = 640;
    int input_height = 640;
    // detection
    float conf_threshold = 0.25f;
    float nms_threshold  = 0.45f;
    int   max_detections = 100;
    // backend
    std::string engine = "opencv";
    std::string target = "cpu";      // cpu | cuda | cuda_fp16
    int num_threads    = 0;
    // alert
    float alert_conf_threshold = 0.6f;
    int   alert_cooldown_ms    = 1500;
    int   alert_banner_ms      = 1500;
    // source
    std::string source = "0";
    bool display = true;
    std::string output_video;
    // benchmark
    int warmup_frames = 20;
    int max_frames    = 300;
    std::string result_json = "outputs/logs/benchmark_cpp.json";
    std::string tag         = "cpp-opencv-dnn";

    // Nem std::runtime_error neu khong mo duoc file.
    static RuntimeConfig load(const std::string& yaml_path);
};

}  // namespace its
