"""Do FPS / do tre cua nhanh Python (de so sanh voi nhanh C++ its_benchmark).

Backend:
  opencv       : cv2.dnn doc best.onnx (cung engine voi C++ -> so sanh cong bang ngon ngu)
  onnxruntime  : onnxruntime doc best.onnx
  pytorch      : ultralytics YOLO('best.pt') - moc "nhanh Python goc" khi huan luyen

Vi du (chay tu thu muc goc du an):
  python scripts/benchmark_python.py --backend opencv      --source data/raw/demo.mp4
  python scripts/benchmark_python.py --backend onnxruntime --source data/raw/demo.mp4
  python scripts/benchmark_python.py --backend pytorch     --source data/raw/demo.mp4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import (Timer, decode_yolo_output, filter_alerts, letterbox,   # noqa: E402
                       load_runtime_config, summarize, to_blob)


class OpenCVBackend:
    name = "opencv-dnn"

    def __init__(self, cfg):
        self.net = cv2.dnn.readNetFromONNX(cfg["model"]["path"])
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.out_names = self.net.getUnconnectedOutLayersNames()

    def infer(self, blob):
        self.net.setInput(blob)
        return self.net.forward(self.out_names)[0]


class OrtBackend:
    name = "onnxruntime"

    def __init__(self, cfg):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(cfg["model"]["path"], providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def infer(self, blob):
        return self.sess.run(None, {self.input_name: blob})[0]


def run(args):
    cfg = load_runtime_config(args.config)
    src = args.source or cfg["source"]["input"]
    w, h = cfg["model"]["input_width"], cfg["model"]["input_height"]
    det, alert = cfg["detection"], cfg["alert"]
    warmup = cfg["benchmark"]["warmup_frames"] if args.warmup is None else args.warmup
    frames = cfg["benchmark"]["max_frames"] if args.frames is None else args.frames

    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"Khong mo duoc nguon video: {src}")

    pt_model = None
    if args.backend == "pytorch":
        from ultralytics import YOLO
        pt_model = YOLO(cfg["model"]["pt_path"])
        backend_name = "pytorch"
    else:
        backend = OpenCVBackend(cfg) if args.backend == "opencv" else OrtBackend(cfg)
        backend_name = backend.name

    pre, inf, post, total, lat = [], [], [], [], []
    decode_sum, measured, wall_t0 = 0.0, 0, None
    for n in range(warmup + frames):
        t_dec = Timer()
        ok, frame = cap.read()
        if not ok:
            if str(src).isdigit():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        dec_ms = t_dec.elapsed_ms()
        if n == warmup:
            wall_t0 = time.perf_counter()

        t_frame = Timer()
        if pt_model is not None:                      # ultralytics tu tach thoi gian cac giai doan
            res = pt_model.predict(frame, imgsz=w, conf=det["conf_threshold"], iou=det["nms_threshold"],
                                   verbose=False)[0]
            sp = res.speed
            t_pre, t_inf, t_post = sp["preprocess"], sp["inference"], sp["postprocess"]
            dets = [(*map(int, b.xyxy[0].tolist()[:2]), 0, 0, float(b.conf), int(b.cls)) for b in res.boxes]
        else:
            t = Timer()
            canvas, scale, pad = letterbox(frame, (w, h))
            blob = to_blob(canvas)
            t_pre = t.elapsed_ms()
            t.reset()
            out = backend.infer(blob)
            t_inf = t.elapsed_ms()
            t.reset()
            dets = decode_yolo_output(out, scale, pad, frame.shape, det["conf_threshold"],
                                      det["nms_threshold"], det["max_detections"])
            t_post = t.elapsed_ms()
        filter_alerts(dets, alert["conf_threshold"])
        latency = t_frame.elapsed_ms()

        if n >= warmup:
            pre.append(t_pre); inf.append(t_inf); post.append(t_post)
            total.append(t_pre + t_inf + t_post); lat.append(latency)
            decode_sum += dec_ms; measured += 1

    if measured == 0:
        sys.exit("Khong du khung hinh de do.")
    wall = time.perf_counter() - wall_t0
    s_total = summarize(total)
    result = {
        "tag": args.tag or f"py-{backend_name}", "language": "Python", "backend": backend_name,
        "target": "cpu", "model": cfg["model"]["pt_path" if pt_model else "path"],
        "input_size": [w, h], "opencv_version": cv2.__version__, "frames": measured,
        "fps_pipeline": 1000.0 / s_total["mean"], "fps_end_to_end": measured / wall,
        "decode_ms_mean": decode_sum / measured,
        "preprocess_ms": summarize(pre), "inference_ms": summarize(inf),
        "postprocess_ms": summarize(post), "total_ms": s_total, "alert_latency_ms": summarize(lat),
    }
    out_path = Path(args.out or f"outputs/logs/benchmark_py_{backend_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{result['tag']}] {measured} khung | FPS(pipeline)={result['fps_pipeline']:.1f} | "
          f"FPS(end-to-end)={result['fps_end_to_end']:.1f} | tre canh bao TB={summarize(lat)['mean']:.2f} ms")
    print(f"Da ghi: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["opencv", "onnxruntime", "pytorch"], default="opencv")
    ap.add_argument("--config", default="configs/its_runtime.yaml")
    ap.add_argument("--source")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--out")
    run(ap.parse_args())
