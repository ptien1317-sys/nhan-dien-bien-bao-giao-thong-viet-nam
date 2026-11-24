// benchmark.cpp - Do FPS va do tre (chay an) va ghi ket qua JSON de so sanh voi nhanh Python.
//   ./its_benchmark --config configs/its_runtime.yaml [--source video.mp4] [--frames 300]
//                   [--warmup 20] [--out outputs/logs/benchmark_cpp.json] [--tag cpp-opencv-dnn]
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/videoio.hpp>

#include "alert.hpp"
#include "detector.hpp"
#include "its_config.hpp"
#include "its_utils.hpp"

int main(int argc, char** argv) {
    std::string cfg_path = "configs/its_runtime.yaml";
    std::string source, out_path, tag, model;
    int frames = -1, warmup = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string& dst) { if (i + 1 < argc) dst = argv[++i]; };
        if (a == "--config") next(cfg_path);
        else if (a == "--source") next(source);
        else if (a == "--out") next(out_path);
        else if (a == "--tag") next(tag);
        else if (a == "--model") next(model);
        else if (a == "--frames" && i + 1 < argc) frames = std::stoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
    }

    try {
        its::RuntimeConfig cfg = its::RuntimeConfig::load(cfg_path);
        if (!source.empty()) cfg.source = source;
        if (!out_path.empty()) cfg.result_json = out_path;
        if (!tag.empty()) cfg.tag = tag;
        if (!model.empty()) cfg.model_path = model;
        if (frames > 0) cfg.max_frames = frames;
        if (warmup >= 0) cfg.warmup_frames = warmup;

        its::Detector detector(cfg);
        its::AlertManager alerts(cfg.alert_conf_threshold, cfg.alert_cooldown_ms,
                                 cfg.alert_banner_ms, detector.labels());
        cv::VideoCapture cap;
        if (!its::openSource(cap, cfg.source)) {
            std::cerr << "Khong mo duoc nguon video: " << cfg.source << "\n";
            return 1;
        }

        std::vector<double> pre, inf, post, total, latency;
        cv::Mat frame;
        const int target = cfg.warmup_frames + cfg.max_frames;
        its::Timer wall;                          // dong ho toan bo (gom ca giai ma video)
        bool wall_started = false;
        double decode_ms_sum = 0;
        long measured = 0;

        for (int n = 0; n < target; ++n) {
            its::Timer dec;
            if (!cap.read(frame)) {               // het video -> tua lai de du so khung hinh
                if (isdigit(static_cast<unsigned char>(cfg.source[0]))) break;   // camera: dung han
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                if (!cap.read(frame)) break;
            }
            const double dec_ms = dec.elapsedMs();
            if (n == cfg.warmup_frames) { wall.reset(); wall_started = true; }

            its::Timer frame_timer;
            its::StageTimings st;
            auto dets = detector.detect(frame, &st);
            alerts.evaluate(dets);
            const double lat = frame_timer.elapsedMs();

            if (n >= cfg.warmup_frames) {
                pre.push_back(st.preprocess_ms); inf.push_back(st.inference_ms);
                post.push_back(st.postprocess_ms); total.push_back(st.total());
                latency.push_back(lat); decode_ms_sum += dec_ms; ++measured;
            }
        }
        if (!wall_started || measured == 0) { std::cerr << "Khong du khung hinh de do.\n"; return 1; }

        const double wall_s = wall.elapsedMs() / 1000.0;
        const its::Summary s_total = its::summarize(total);
        std::ofstream js(cfg.result_json);
        if (!js) { std::cerr << "Khong ghi duoc " << cfg.result_json << "\n"; return 1; }
        js << "{\n"
           << "  \"tag\": \"" << cfg.tag << "\",\n"
           << "  \"language\": \"C++\",\n"
           << "  \"backend\": \"opencv-dnn\",\n"
           << "  \"target\": \"" << cfg.target << "\",\n"
           << "  \"model\": \"" << cfg.model_path << "\",\n"
           << "  \"input_size\": [" << cfg.input_width << ", " << cfg.input_height << "],\n"
           << "  \"opencv_version\": \"" << CV_VERSION << "\",\n"
           << "  \"frames\": " << measured << ",\n"
           << "  \"fps_pipeline\": " << (s_total.mean > 0 ? 1000.0 / s_total.mean : 0) << ",\n"
           << "  \"fps_end_to_end\": " << measured / wall_s << ",\n"
           << "  \"decode_ms_mean\": " << decode_ms_sum / measured << ",\n"
           << "  \"preprocess_ms\": " << its::toJson(its::summarize(pre)) << ",\n"
           << "  \"inference_ms\": " << its::toJson(its::summarize(inf)) << ",\n"
           << "  \"postprocess_ms\": " << its::toJson(its::summarize(post)) << ",\n"
           << "  \"total_ms\": " << its::toJson(s_total) << ",\n"
           << "  \"alert_latency_ms\": " << its::toJson(its::summarize(latency)) << "\n"
           << "}\n";

        std::cout << "[" << cfg.tag << "] " << measured << " khung hinh | "
                  << "FPS(pipeline)=" << 1000.0 / s_total.mean << " | FPS(end-to-end)=" << measured / wall_s
                  << " | tre canh bao TB=" << its::summarize(latency).mean << " ms\n"
                  << "Da ghi: " << cfg.result_json << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Loi: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
