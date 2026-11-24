"""Chạy toàn bộ nhiệm vụ 2 bằng MỘT lệnh:  python run_all.py
Việc làm: dọn data/processed -> tiền xử lý + chia tập -> tăng cường -> vẽ 2 hình -> in số liệu cho báo cáo."""
import json, random, re, shutil, subprocess, sys
from pathlib import Path

import yaml

CFG = "configs/dataset.yaml"
root = Path("data/processed")
AUG = lambda stem: re.search(r"_aug\d+$", stem) is not None


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True)


if root.exists():
    for p in root.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
run("src.data_pipeline.prepare_dataset", "--config", CFG)
run("src.data_pipeline.augment_dataset", "--config", CFG)
run("src.data_pipeline.class_report", "--config", CFG)

originals = sorted(p for p in (root / "train" / "images").glob("*.jpg") if not AUG(p.stem))
sample = random.Random(1).choice([p for p in originals if p.stat().st_size > 3000] or originals)
run("src.data_pipeline.visualize", "augment", "--image", str(sample))

print("\n===== SỐ LIỆU ĐIỀN VÀO BÁO CÁO (mục 2.3.1 c) =====")
for s in ("train", "val", "test"):
    files = list((root / s / "images").glob("*.jpg"))
    goc = sum(not AUG(f.stem) for f in files)
    print(f"{s:6s} ảnh gốc: {goc:6d} | tổng sau tăng cường: {len(files):6d}")
st = json.load(open(root / "stats.json"))
print("Theo nguồn:", {k: dict(v) for k, v in st["source"].items()})
cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
nc, names = cfg["nc"], cfg["names"]
pc = {c: sum(st["per_class"].get(s, {}).get(str(c), 0) for s in ("train", "val", "test")) for c in range(nc)}
top = sorted(pc.items(), key=lambda kv: -kv[1])
print(f"Số lớp: {nc}")
print("5 lớp nhiều ảnh nhất:", [(c, names[c], n) for c, n in top[:5]])
print("5 lớp ít ảnh nhất  :", [(c, names[c], n) for c, n in top[-5:]])
print("\nHình đã lưu ở outputs/figures/. Gửi cho mình: data/processed/stats.json + 2 hình PNG + dòng in ở trên.")
