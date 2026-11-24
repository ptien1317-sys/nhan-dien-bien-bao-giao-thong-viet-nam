"""Sinh outputs/checkpoints/labels.txt tu configs/dataset.yaml (Thanh vien 2) cho nhanh C++.

dataset.yaml can co khoa `names` dang list hoac dict {id: ten} (chuan Ultralytics).
Chay:  python -m src.utils.export_labels --dataset configs/dataset.yaml
"""
import argparse
from pathlib import Path
import yaml


def export_labels(dataset_yaml: str, out_path: str) -> int:
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(n) for n in names) + "\n", encoding="utf-8")
    return len(names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="configs/dataset.yaml")
    ap.add_argument("--out", default="outputs/checkpoints/labels.txt")
    a = ap.parse_args()
    print(f"Da ghi {export_labels(a.dataset, a.out)} lop -> {a.out}")
