"""Vẽ hình minh họa cho báo cáo: (1) kết quả Augmentation, (2) thống kê bộ dữ liệu.

Chạy:
    python -m src.data_pipeline.visualize augment --image <ảnh> --out outputs/figures/augmentation_examples.png
    python -m src.data_pipeline.visualize stats --config configs/dataset.yaml --out outputs/figures/dataset_distribution.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .augmentation import EFFECTS
from .preprocess import preprocess_image

TITLES = {"rain": "Mưa", "fog": "Sương mù", "motion_blur": "Nhòe chuyển động",
          "low_light": "Thiếu sáng", "glare": "Chói đèn pha", "noise": "Nhiễu cảm biến"}


def figure_augmentation(image: str, out: str, seed: int = 7) -> None:
    img = cv2.imread(image)
    rng = np.random.default_rng(seed)
    panels = [("Ảnh gốc", img), ("Sau tiền xử lý\n(CLAHE + lọc song phương)",
              preprocess_image(img, {"clahe": True, "denoise": "bilateral"}))]
    panels += [(TITLES[k], fn(img, rng)) for k, fn in EFFECTS.items()]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, (title, im) in zip(axes.ravel(), panels):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Đã lưu", out)


def figure_stats(config: str, out: str) -> None:
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    stats = json.load(open(Path(cfg["path"]) / "stats.json"))
    nc = cfg["nc"]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    bottom = np.zeros(nc)
    for split, color in (("train", "#2b6cb0"), ("val", "#dd8b2d"), ("test", "#2f9e6b")):
        vals = np.array([stats["per_class"].get(split, {}).get(str(c), 0) for c in range(nc)])
        ax.bar(range(nc), vals, bottom=bottom, label=f"{split} ({int(vals.sum())})", color=color)
        bottom += vals
    ax.set_xlabel("Mã lớp biển báo (ClassId)")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố số ảnh theo lớp và theo tập dữ liệu")
    ax.set_xticks(range(nc))
    ax.tick_params(axis="x", labelsize=7)
    ax.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Đã lưu", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("augment")
    a.add_argument("--image", required=True)
    a.add_argument("--out", default="outputs/figures/augmentation_examples.png")
    s = sub.add_parser("stats")
    s.add_argument("--config", default="configs/dataset.yaml")
    s.add_argument("--out", default="outputs/figures/dataset_distribution.png")
    args = ap.parse_args()
    figure_augmentation(args.image, args.out) if args.cmd == "augment" else figure_stats(args.config, args.out)


if __name__ == "__main__":
    main()
