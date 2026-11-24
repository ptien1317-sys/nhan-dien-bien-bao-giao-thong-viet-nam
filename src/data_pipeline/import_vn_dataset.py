"""Nhập bộ dữ liệu biển báo Việt Nam định dạng YOLO vào data/raw/vietnam/ và cập nhật configs/dataset.yaml.
Hỗ trợ: bản export Roboflow (YOLOv8/YOLO11: train/valid/test + data.yaml) và bộ Kaggle (images/labels/classes.txt).

Chạy:  python -m src.data_pipeline.import_vn_dataset --src "C:\\vn_signs"
  --append   giữ dữ liệu đã nhập trước đó (dùng khi muốn gộp nhiều bộ; nhớ các bộ phải cùng bảng lớp)
Mặc định: xoá dữ liệu cũ trong data/raw/vietnam rồi nhập lại.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SKIP_TXT = {"classes.txt", "notes.txt", "readme.txt", "labels.txt", "readme.dataset.txt", "readme.roboflow.txt"}
SPLIT_ALIAS = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}


def read_names(src: Path) -> list[str]:
    for y in list(src.rglob("data.yaml")) + list(src.rglob("*.yaml")):
        d = yaml.safe_load(open(y, encoding="utf-8")) or {}
        n = d.get("names")
        if isinstance(n, dict):
            return [str(n[k]) for k in sorted(n, key=int)]
        if isinstance(n, list):
            return [str(x) for x in n]
    cls = next(iter(src.rglob("classes.txt")), None)
    if cls is not None:
        return [l.strip() for l in cls.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise SystemExit("Không tìm thấy data.yaml (có 'names') hoặc classes.txt trong thư mục nguồn.")


def split_of(img: Path, src: Path) -> str:
    for part in img.relative_to(src).parts[:-1]:
        if part.lower() in SPLIT_ALIAS:
            return SPLIT_ALIAS[part.lower()]
    return ""


def find_label(img: Path, fallback: dict) -> Path | None:
    sib = Path(str(img.parent).replace("images", "labels")) / f"{img.stem}.txt"   # .../train/images -> .../train/labels
    if sib.exists():
        return sib
    return fallback.get(img.stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thư mục đã giải nén của bộ dữ liệu")
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Không thấy thư mục: {src}")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = Path(cfg["raw"]["custom"])
    if out.exists() and not args.append:
        shutil.rmtree(out)
    for sub in ("images", "labels"):                 # nhập bộ mới = thay thế bộ cũ, tránh trộn lẫn
        shutil.rmtree(out / sub, ignore_errors=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    names = read_names(src)
    fallback = {p.stem: p for p in src.rglob("*.txt") if p.name.lower() not in SKIP_TXT}
    n_ok = n_nolabel = bad = 0
    used, per_class, per_split, rows = set(), Counter(), Counter(), []
    sp_file = out / "splits.csv"
    if args.append and sp_file.exists():
        rows = [(r["stem"], r["split"]) for r in csv.DictReader(open(sp_file, encoding="utf-8"))]
        used = {r[0] for r in rows}
    for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT):
        if "meta" in [x.lower() for x in img.relative_to(src).parts[:-1]]:
            continue
        lbl = find_label(img, fallback)
        if lbl is None:
            n_nolabel += 1
            continue
        stem, k = img.stem, 1
        while stem in used:
            stem, k = f"{img.stem}_dup{k}", k + 1
        lines = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[0].isdigit() and int(parts[0]) < len(names):
                lines.append(line.strip())
                per_class[int(parts[0])] += 1
            elif parts:
                bad += 1
        if not lines:
            continue
        used.add(stem)
        sp = split_of(img, src)
        rows.append((stem, sp))
        per_split[sp or "(không rõ)"] += 1
        shutil.copy(img, out / "images" / f"{stem}{img.suffix.lower()}")
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_ok += 1

    has_presets = bool(rows) and all(sp in ("train", "val", "test") for _, sp in rows) and \
        {"train", "val", "test"} <= {sp for _, sp in rows}
    if has_presets:
        with open(sp_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["stem", "split"]); w.writerows(rows)
    elif sp_file.exists():
        sp_file.unlink()

    roboflow_aug = any(".rf." in s for s, _ in rows)
    cfg["raw"].setdefault("gtsrb", "data/raw/gtsrb")   # GTSRB bật/tắt bằng gtsrb_merge.enabled
    cfg["raw"].setdefault("own", "data/raw/own")
    cfg["nc"] = len(names)
    cfg["names"] = {i: n for i, n in enumerate(names)}
    cfg["preprocess"]["max_side"] = 0
    cfg["split"]["mode"] = "keep" if has_presets else "random"
    # bộ Roboflow có ảnh tăng cường sẵn: gom bản tăng cường theo ảnh gốc, val/test chỉ giữ 1 ảnh/nhóm
    cfg["split"]["custom_group_by"] = "roboflow" if roboflow_aug else "image"
    if roboflow_aug:
        cfg["augmentation"]["copies_per_image"] = 1      # bản Roboflow đã có ảnh augment sẵn -> bớt nhân bản thêm
    yaml.safe_dump(cfg, open(args.config, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)

    print(f"Đã nhập {n_ok} ảnh có nhãn vào {out} ({n_nolabel} ảnh không có nhãn bị bỏ, {bad} dòng nhãn lỗi bị bỏ).")
    print(f"Số ảnh theo tập trong bộ gốc: {dict(per_split)}")
    print(f"Chế độ chia tập: {cfg['split']['mode']} | số lớp: {len(names)} (đã ghi vào {args.config})")
    if roboflow_aug:
        print("Phát hiện ảnh dạng Roboflow (.rf.) -> các bản augment của cùng ảnh gốc được giữ chung một tập.")
    print("Số khung biển theo lớp (id: tên = số lượng):")
    for cid in range(len(names)):
        print(f"  {cid:2d}: {names[cid]} = {per_class.get(cid, 0)}")
    empty = [names[c] for c in range(len(names)) if per_class.get(c, 0) == 0]
    if empty:
        print("[CẢNH BÁO] Lớp không có nhãn nào:", empty)


if __name__ == "__main__":
    main()
