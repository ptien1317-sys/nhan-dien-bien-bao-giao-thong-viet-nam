"""Báo cáo phân bố lớp CHÍNH XÁC theo từng khung nhãn (không chỉ khung đầu tiên của mỗi ảnh).

Chạy:  python -m src.data_pipeline.class_report --config configs/dataset.yaml
Đầu ra:
    - in ra màn hình: số ảnh / số ảnh gốc duy nhất / số khung nhãn mỗi tập, cảnh báo lớp hiếm
    - outputs/class_report.csv            (bảng từng lớp x từng tập, mở được bằng Excel)
    - outputs/figures/dataset_distribution.png   (hình cho báo cáo)
Chỉ đếm ảnh GỐC (bỏ ảnh tăng cường *_aug<k>) để số liệu phản ánh đúng dữ liệu thật.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

AUG = re.compile(r"_aug\d+$")
SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--rare", type=int, default=20, help="ngưỡng số khung nhãn coi là lớp hiếm")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root, nc, names = Path(cfg["path"]), cfg["nc"], cfg["names"]

    boxes = {s: Counter() for s in SPLITS}
    n_img = Counter()
    n_aug = Counter()
    for s in SPLITS:
        for lbl in (root / s / "labels").glob("*.txt"):
            if AUG.search(lbl.stem):
                n_aug[s] += 1
                continue
            n_img[s] += 1
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    boxes[s][int(parts[0])] += 1

    groups = defaultdict(set)                       # số ảnh gốc duy nhất (gom các bản Roboflow cùng ảnh gốc)
    mf = root / "manifest.csv"
    if mf.exists():
        for r in csv.DictReader(open(mf, encoding="utf-8")):
            if not AUG.search(Path(r["file"]).stem):
                groups[r["split"]].add(r["group"])

    print("===== TỔNG QUAN (chỉ tính ảnh gốc, không tính ảnh tăng cường của pipeline) =====")
    for s in SPLITS:
        print(f"{s:6s} ảnh: {n_img[s]:6d} | ảnh gốc duy nhất: {len(groups[s]):6d} | "
              f"khung nhãn: {sum(boxes[s].values()):6d} | ảnh tăng cường thêm: {n_aug[s]}")

    rows = []
    for c in range(nc):
        tr, va, te = (boxes[s][c] for s in SPLITS)
        rows.append((c, names[c], tr, va, te, tr + va + te))
    out_csv = Path("outputs/class_report.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "ten_lop", "train", "val", "test", "tong_khung_nhan"])
        w.writerows(rows)

    total = sorted(rows, key=lambda r: -r[5])
    nonzero = [r[5] for r in rows if r[5] > 0]
    print(f"\n===== PHÂN BỐ THEO LỚP ({nc} lớp, tổng {sum(r[5] for r in rows)} khung nhãn) =====")
    print("10 lớp nhiều nhất:", [(r[1], r[5]) for r in total[:10]])
    print("10 lớp ít nhất   :", [(r[1], r[5]) for r in total[-10:]])
    if nonzero:
        print(f"Tỷ lệ mất cân bằng (nhiều nhất / ít nhất): {max(nonzero) / min(nonzero):.0f} lần")
    rare = [(r[1], r[5]) for r in rows if r[5] < args.rare]
    print(f"\nLớp hiếm (< {args.rare} khung nhãn): {len(rare)} lớp -> {rare}")
    no_val = [r[1] for r in rows if r[3] == 0 and r[5] > 0]
    no_test = [r[1] for r in rows if r[4] == 0 and r[5] > 0]
    print(f"Lớp không có trong tập val ({len(no_val)}): {no_val}")
    print(f"Lớp không có trong tập test ({len(no_test)}): {no_test}")
    print(f"\nĐã lưu bảng chi tiết: {out_csv}")

    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ["#c0392b" if r[5] < args.rare else "#2b6cb0" for r in total]
    ax.bar(range(nc), [r[5] for r in total], color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(nc))
    ax.set_xticklabels([r[1] for r in total], rotation=90, fontsize=7)
    ax.set_ylabel("Số khung nhãn (thang log)")
    ax.set_title(f"Phân bố số khung nhãn theo lớp (đỏ: lớp hiếm, < {args.rare} khung)")
    fig.tight_layout()
    fig_path = Path("outputs/figures/dataset_distribution.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Đã lưu hình: {fig_path}")


if __name__ == "__main__":
    main()
