// main.cpp - Ung dung suy luan thoi gian thuc: video/camera -> Detector -> Alert -> hien thi/ghi video
#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "alert.hpp"
#include "detector.hpp"
#include "its_config.hpp"
#include "its_utils.hpp"

static void usage(const char* prog) {
    std::cout << "Cach dung: " << prog << " [--config configs/its_runtime.yaml] [--source <video|camera_id>]\n"
              << "           [--model best.onnx] [--output out.mp4] [--no-display]\n";
}

int main(int argc, char** argv) {
    std::string cfg_path = "configs/its_runtime.yaml";
    std::string source_override, model_override, output_override;
    bool no_display = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string& dst) { if (i + 1 < argc) dst = argv[++i]; };
        if (a == "--config") next(cfg_path);
        else if (a == "--source") next(source_override);
        else if (a == "--model") next(model_override);
        else if (a == "--output") next(output_override);
        else if (a == "--no-display") no_display = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    }

    try {
        its::RuntimeConfig cfg = its::RuntimeConfig::load(cfg_path);
        if (!source_override.empty()) cfg.source = source_override;
        if (!model_override.empty()) cfg.model_path = model_override;
        if (!output_override.empty()) cfg.output_video = output_override;
        if (no_display) cfg.display = false;

        its::Detector detector(cfg);
        its::AlertManager alerts(cfg.alert_conf_threshold, cfg.alert_cooldown_ms,
                                 cfg.alert_banner_ms, detector.labels());

        cv::VideoCapture cap;
        if (!its::openSource(cap, cfg.source)) {
            std::cerr << "Khong mo duoc nguon video: " << cfg.source << "\n";
            return 1;
        }
        cv::VideoWriter writer;
        const double src_fps = cap.get(cv::CAP_PROP_FPS) > 1 ? cap.get(cv::CAP_PROP_FPS) : 30.0;

        detector.warmup(3);
        std::vector<double> total_ms, latency_ms;
        double fps_smooth = 0;
        cv::Mat frame;
        long n = 0;

        while (cap.read(frame)) {
            its::Timer frame_timer;                              // moc t0: ngay khi co khung hinh
            its::StageTimings st;
            auto dets = detector.detect(frame, &st);
            auto new_alerts = alerts.evaluate(dets);
            const double latency = frame_timer.elapsedMs();      // t1: tin hieu canh bao da san sang
            total_ms.push_back(st.total());
            latency_ms.push_back(latency);

            for (const auto& a : new_alerts)
                std::cout << "[ALERT] frame " << n << ": " << a.message << " (conf=" << a.confidence
                          << ", tre=" << latency << " ms)\n";

            const double inst = 1000.0 / std::max(1e-3, frame_timer.elapsedMs());
            fps_smooth = (n == 0) ? inst : 0.9 * fps_smooth + 0.1 * inst;
            alerts.draw(frame, dets, fps_smooth);

            if (!cfg.output_video.empty()) {
                if (!writer.isOpened())
                    writer.open(cfg.output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), src_fps, frame.size());
                writer.write(frame);
            }
            if (cfg.display) {
                cv::imshow("ITS - Nhan dien bien bao (q/ESC de thoat)", frame);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q') break;
            }
            ++n;
        }

        const its::Summary s = its::summarize(total_ms), l = its::summarize(latency_ms);
        std::cout << "\n=== Tong ket (" << n << " khung hinh) ===\n"
                  << "Thoi gian xu ly TB : " << s.mean << " ms/frame  (~" << (s.mean > 0 ? 1000.0 / s.mean : 0) << " FPS)\n"
                  << "Do tre canh bao TB : " << l.mean << " ms (p95 = " << l.p95 << " ms)\n";
    } catch (const std::exception& e) {
        std::cerr << "Loi: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
