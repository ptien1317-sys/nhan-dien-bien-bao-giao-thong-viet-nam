import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from src.data_pipeline.preprocess import balance_brightness as apply_clahe_lighting
from src.inference.alert_manager import ITSAlertManager
from src.data_pipeline.preprocess import preprocess_image

def apply_clahe_lighting(img):
    return preprocess_image(img, {"clahe": True, "clahe_clip_limit": 2.0, "clahe_tile_grid": 8, "denoise": "bilateral"})


class TrafficSignPredictor:
    def __init__(self, weights_path: str = "outputs/checkpoints/best.pt", conf_thres: float = 0.25):
        if os.path.exists(weights_path):
            self.model_path = weights_path
        else:
            self.model_path = "yolov8n.pt"

        self.model = YOLO(self.model_path)
        self.conf_thres = conf_thres
        self.alert_manager = ITSAlertManager()

    def predict_frame(self, frame_bgr: np.ndarray, use_clahe: bool = False, is_static_image: bool = False):
        t0 = time.perf_counter()
        proc_frame = apply_clahe_lighting(frame_bgr) if use_clahe else frame_bgr.copy()
        orig_h, orig_w = proc_frame.shape[:2]

        # 1. Thử nhận diện trực tiếp trên ảnh gốc (Dành cho ảnh đường phố, video hành trình, webcam ở khoảng cách 40cm)
        results = self.model.predict(proc_frame, conf=self.conf_thres, verbose=False)[0]
        used_canvas = False

        # 2. Nếu là ảnh biển báo cắt sát viền (hoặc dí quá sát camera làm biển báo tràn 90% khung hình):
        # Tự động thu nhỏ biển báo đặt lên nền trung tính ở các tỷ lệ 40%, 25%, 15% giống kích thước ngoài đường thật
        if results.boxes is None or len(results.boxes) == 0:
            best_box_res = None
            best_conf = 0.0
            for scale_ratio in [0.40, 0.25, 0.16]:
                canvas_size = 640
                # Nền xám xanh nhẹ mô phỏng bầu trời/mặt đường ngoài trời
                canvas = np.full((canvas_size, canvas_size, 3), (185, 190, 185), dtype=np.uint8)
                target_w = max(36, int(canvas_size * scale_ratio))
                target_h = max(36, int(orig_h * (target_w / max(orig_w, 1))))
                if target_h > int(canvas_size * 0.65):
                    target_h = int(canvas_size * scale_ratio)
                    target_w = max(36, int(orig_w * (target_h / max(orig_h, 1))))

                resized_sign = cv2.resize(proc_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                y_off = (canvas_size - target_h) // 2
                x_off = (canvas_size - target_w) // 2
                canvas[y_off:y_off + target_h, x_off:x_off + target_w] = resized_sign

                cand_res = self.model.predict(canvas, conf=0.15, verbose=False)[0]
                if cand_res.boxes is not None and len(cand_res.boxes) > 0:
                    max_c = float(cand_res.boxes.conf[0].item())
                    if max_c > best_conf:
                        best_conf = max_c
                        best_box_res = cand_res

            if best_box_res is not None:
                results = best_box_res
                used_canvas = True

        annotated_frame = frame_bgr.copy() if used_canvas else results.plot()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-3)

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label_name = results.names.get(cls_id, str(cls_id))
                packet = self.alert_manager.process_detection(cls_id, label_name, conf)
                detections.append(packet)

                if used_canvas:
                    cv2.rectangle(annotated_frame, (8, 8), (orig_w - 8, orig_h - 8), (0, 230, 0), 4)
                    cv2.putText(annotated_frame, f"{packet['code']} ({conf*100:.1f}%)",
                                (16, max(40, orig_h - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        cv2.rectangle(annotated_frame, (8, 8), (240, 42), (20, 30, 45), -1)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} ({latency_ms:.0f}ms)", (14, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)

        return annotated_frame, detections, round(fps, 1), round(latency_ms, 1)
