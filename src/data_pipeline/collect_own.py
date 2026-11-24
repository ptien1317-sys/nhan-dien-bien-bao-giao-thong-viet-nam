"""Công cụ thu thập dữ liệu tự quay của nhóm (video -> khung hình -> gán nhãn YOLO).

1) Tạo danh sách lớp để nạp vào công cụ gán nhãn (LabelImg / Roboflow / CVAT):
       python -m src.data_pipeline.collect_own classes
   -> data/raw/own/classes.txt (đúng thứ tự mã lớp trong configs/dataset.yaml)

2) Tách khung hình từ video (mặc định 1 khung / giây):
       python -m src.data_pipeline.collect_own frames --video "D:\\video\\quay1.mp4" --name quay1
   -> data/raw/own/images/quay1_0000.jpg, quay1_0001.jpg ...
   Tên khung hình dạng <tên_video>_<số>: pipeline gom cùng video vào cùng một tập, tránh rò rỉ dữ liệu.

3) Gán nhãn bằng LabelImg (định dạng YOLO, lưu file .txt vào data/raw/own/labels/).
4) Chạy:  python run_all.py   (tự gộp data/raw/own/ vào bộ dữ liệu).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml


def cmd_classes(args) -> None:
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = Path(cfg["raw"].get("own") or "data/raw/own")
    out.mkdir(parents=True, exist_ok=True)
    names = [cfg["names"][i] for i in range(cfg["nc"])]
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Đã ghi {len(names)} lớp vào {out / 'classes.txt'}")


def cmd_frames(args) -> None:
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = Path(cfg["raw"].get("own") or "data/raw/own") / "images"
    (out.parent / "labels").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * args.every_sec))
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(str(out / f"{args.name}_{n:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
        i += 1
    cap.release()
    print(f"Đã lưu {n} khung hình (mỗi {args.every_sec}s) vào {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classes")
    c.add_argument("--config", default="configs/dataset.yaml")
    f = sub.add_parser("frames")
    f.add_argument("--video", required=True)
    f.add_argument("--name", required=True, help="tên ngắn của video, vd. quay1")
    f.add_argument("--every-sec", type=float, default=1.0)
    f.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cmd_classes(args) if args.cmd == "classes" else cmd_frames(args)


if __name__ == "__main__":
    main()
