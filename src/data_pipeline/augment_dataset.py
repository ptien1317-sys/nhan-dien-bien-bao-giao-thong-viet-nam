"""Sinh thêm ảnh tăng cường (mưa, sương mù, motion blur...) CHỈ cho tập train.
Nhãn được sao chép nguyên vẹn vì các hiệu ứng không làm thay đổi hình học.

Chạy:  python -m src.data_pipeline.augment_dataset --config configs/dataset.yaml
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .augmentation import augment_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    acfg = cfg["augmentation"]
    rng = np.random.default_rng(acfg.get("seed", 42))

    train = Path(cfg["path"]) / "train"
    imgs = sorted(p for p in (train / "images").glob("*.jpg") if not re.search(r"_aug\d+$", p.stem))
    log = []
    for p in imgs:
        img = cv2.imread(str(p))
        lbl = train / "labels" / f"{p.stem}.txt"
        for k in range(int(acfg["copies_per_image"])):
            out, applied = augment_image(img, rng, acfg["prob"])
            name = f"{p.stem}_aug{k}"
            cv2.imwrite(str(train / "images" / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy(lbl, train / "labels" / f"{name}.txt")
            log.append((f"{name}.jpg", p.name, "+".join(applied)))
    with open(train / "augmentation_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "origin", "effects"])
        w.writerows(log)
    print(f"Đã sinh {len(log)} ảnh tăng cường từ {len(imgs)} ảnh gốc của tập train.")


if __name__ == "__main__":
    main()
