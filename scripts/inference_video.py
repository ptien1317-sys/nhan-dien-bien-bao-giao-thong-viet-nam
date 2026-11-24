"""Chạy nhận diện trên video và xuất kết quả lưu vào outputs/demo_videos/ (Nhiệm vụ Thành viên 4)."""
import sys
import os
import argparse
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.predictor import TrafficSignPredictor

def process_video(input_video: str, output_video: str = "outputs/demo_videos/demo_result.mp4", conf: float = 0.30):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    predictor = TrafficSignPredictor("outputs/checkpoints/best.pt", conf_thres=conf)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {input_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_frame, _, _, _ = predictor.predict_frame(frame)
        writer.write(out_frame)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"[TV4 Video Exporter] Đã xử lý {frame_count} khung hình -> Lưu tại: {output_video}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Đường dẫn video đầu vào")
    ap.add_argument("--out", default="outputs/demo_videos/demo_result.mp4")
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()
    process_video(args.source, args.out, args.conf)