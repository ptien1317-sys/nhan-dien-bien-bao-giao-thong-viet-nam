"""Gom dữ liệu thô (GTSRB + ảnh thực tế VN) -> tiền xử lý -> chia train/val/test -> data/processed/.

Đầu ra (định dạng YOLO, dùng trực tiếp để huấn luyện):
    data/processed/{train,val,test}/images/*.jpg
    data/processed/{train,val,test}/labels/*.txt     # class_id xc yc w h (chuẩn hóa 0-1)
    data/processed/manifest.csv                      # mỗi ảnh thuộc tập nào, nguồn, lớp, nhóm (track)
    data/processed/stats.json                        # thống kê số ảnh theo tập / theo lớp

Chạy:  python -m src.data_pipeline.prepare_dataset --config configs/dataset.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

from .preprocess import preprocess_image

IMG_EXT = {".ppm", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Sample:
    img_path: Path
    source: str                                  # "gtsrb" | "vietnam"
    group: str                                   # đơn vị chia tập (track GTSRB / video quay)
    boxes: list = field(default_factory=list)    # [(class_id, xc, yc, w, h)] chuẩn hóa
    stratum: int = 0                             # lớp dùng để chia phân tầng
    uid: str = ""                                # tên file đầu ra, duy nhất toàn bộ dữ liệu
    preset: str = ""                             # tập đã định sẵn (vd. từ bản export Roboflow): train|val|test


def _gtsrb_sample(img_path: Path, cid: int, w: float, h: float, x1, y1, x2, y2) -> Sample:
    """Tên file GTSRB có dạng <track>_<frame> (bản gốc) hoặc <lớp>_<track>_<frame> (bản Kaggle)."""
    parts = img_path.stem.split("_")
    track = parts[-2]                                   # ~30 ảnh liên tiếp của cùng một biển vật lý
    box = (cid, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
    uid = f"gtsrb_{cid:05d}_{img_path.stem}"            # kèm mã lớp để tên file không bao giờ trùng
    return Sample(img_path, "gtsrb", f"gtsrb_{cid:05d}_{track}", [box], cid, uid)


def collect_gtsrb(root: Path) -> list[Sample]:
    """Hỗ trợ 2 dạng GTSRB:
    (A) Kaggle: <root>/Train.csv (cột Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path) + <root>/Train/<0..42>/*.png
    (B) Bản gốc INI: <root>/<00000..00042>/*.ppm + GT-xxxxx.csv (sep=';')
    Chỉ dùng tập Train (có thông tin track để chia không rò rỉ)."""
    samples: list[Sample] = []
    if not root.exists():
        return samples

    train_csv = next(iter([root / "Train.csv"] + list(root.rglob("Train.csv"))), None)
    if train_csv is not None and train_csv.exists():           # (A) Kaggle
        base = train_csv.parent
        with open(train_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img = base / row["Path"].replace("\\", "/")
                if not img.exists():
                    continue
                samples.append(_gtsrb_sample(
                    img, int(row["ClassId"]), float(row["Width"]), float(row["Height"]),
                    float(row["Roi.X1"]), float(row["Roi.Y1"]), float(row["Roi.X2"]), float(row["Roi.Y2"])))
        return samples

    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):   # (B) bản gốc
        csvs = list(cls_dir.glob("GT-*.csv"))
        if not csvs:
            continue
        with open(csvs[0], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                samples.append(_gtsrb_sample(
                    cls_dir / row["Filename"], int(row["ClassId"]), float(row["Width"]), float(row["Height"]),
                    *(float(row[k]) for k in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"))))
    return samples


def collect_custom(root: Path, group_by: str = "image", source: str = "vietnam", prefix: str = "vn") -> list[Sample]:
    """Ảnh Việt Nam: images/*.jpg + labels/*.txt (YOLO). Tuỳ chọn splits.csv (stem,split) lưu tập gốc của bản export Roboflow.
    Nhóm (đơn vị chia tập):
      - Roboflow: '<tên_gốc>_jpg.rf.<mã>' -> nhóm = <tên_gốc> (các bản augment của cùng 1 ảnh luôn đi cùng nhau)
      - group_by='prefix': '<tên_video>_<số_khung>' -> nhóm = <tên_video>
      - mặc định: mỗi ảnh một nhóm."""
    samples: list[Sample] = []
    img_dir, lbl_dir = root / "images", root / "labels"
    if not img_dir.exists():
        return samples
    presets = {}
    sp_file = root / "splits.csv"
    if sp_file.exists():
        with open(sp_file, newline="", encoding="utf-8") as f:
            presets = {r["stem"]: r["split"] for r in csv.DictReader(f)}
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            print(f"[CẢNH BÁO] Bỏ qua {p.name}: chưa có file nhãn")
            continue
        boxes = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), *map(float, parts[1:])))
        if not boxes:
            continue
        if ".rf." in p.stem:
            group = f"{prefix}_" + p.stem.split(".rf.")[0]
        elif group_by == "prefix":
            group = f"{prefix}_" + (p.stem.rsplit("_", 1)[0] if "_" in p.stem else p.stem)
        else:
            group = f"{prefix}_" + p.stem
        samples.append(Sample(p, source, group, boxes, boxes[0][0], f"{prefix}_{p.stem}", presets.get(p.stem, "")))
    return samples


def select_gtsrb(samples: list[Sample], id_map: dict[int, int], cap: int, seed: int) -> list[Sample]:
    """Chỉ giữ các lớp GTSRB có ánh xạ sang lớp Việt Nam, đổi mã lớp, và giới hạn tối đa `cap` ảnh/lớp đích
    (lấy trọn từng track để không tách các ảnh gần giống nhau)."""
    rng = random.Random(seed)
    tracks: dict[tuple[int, str], list[Sample]] = defaultdict(list)
    for s in samples:
        if s.stratum in id_map:
            tracks[(id_map[s.stratum], s.group)].append(s)
    by_target: dict[int, list[list[Sample]]] = defaultdict(list)
    for (tgt, _), lst in sorted(tracks.items()):
        by_target[tgt].append(lst)
    out: list[Sample] = []
    for tgt, groups in sorted(by_target.items()):
        rng.shuffle(groups)
        n = 0
        for lst in groups:
            if n >= cap:
                break
            for s in lst:
                s.boxes = [(tgt,) + tuple(b[1:]) for b in s.boxes]
                s.stratum = tgt
                s.preset = ""
                out.append(s)
            n += len(lst)
    return out


def split_groups(samples: list[Sample], ratios: dict, seed: int, by_group: bool = True) -> dict[int, str]:
    """Chia theo NHÓM, phân tầng theo lớp. Trả về {index mẫu: tên tập}."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s.group if by_group else f"{s.group}#{i}"].append(i)
    by_class: dict[int, list[str]] = defaultdict(list)
    for g, idxs in groups.items():
        by_class[samples[idxs[0]].stratum].append(g)

    assign: dict[int, str] = {}
    t, v = ratios["train"], ratios["val"]
    for cid, glist in sorted(by_class.items()):
        glist.sort()
        rng.shuffle(glist)
        n = len(glist)
        n_train = max(1, round(n * t)) if n >= 3 else n
        n_val = max(1, round(n * v)) if n >= 3 else 0
        for k, g in enumerate(glist):
            name = "train" if k < n_train else "val" if k < n_train + n_val else "test"
            for i in groups[g]:
                assign[i] = name
    return assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--no-preprocess", action="store_true", help="không áp dụng CLAHE/lọc nhiễu khi ghi ảnh")
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N ảnh (để chạy thử nhanh)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_root = Path(cfg["path"])
    raw = cfg["raw"]
    samples: list[Sample] = []
    gm = cfg.get("gtsrb_merge") or {}
    if gm.get("enabled") and raw.get("gtsrb"):                      # kết hợp GTSRB qua bảng ánh xạ lớp
        name2id = {str(n): int(i) for i, n in cfg["names"].items()}
        id_map = {int(g): name2id[v] for g, v in (gm.get("map") or {}).items() if v in name2id}
        for g, v in (gm.get("map") or {}).items():
            if v not in name2id:
                print(f"[CẢNH BÁO] gtsrb_merge: lớp '{v}' không có trong danh sách lớp, bỏ qua ánh xạ GTSRB {g}")
        gt = select_gtsrb(collect_gtsrb(Path(raw["gtsrb"])), id_map, int(gm.get("max_per_class", 150)), cfg["split"]["seed"])
        print(f"GTSRB: thêm {len(gt)} ảnh thuộc {len({x.stratum for x in gt})} lớp (tối đa {gm.get('max_per_class', 150)} ảnh/lớp)")
        samples += gt
    samples += collect_custom(Path(raw["custom"]), cfg["split"].get("custom_group_by", "image"))
    if raw.get("own") and (Path(raw["own"]) / "images").exists():     # ảnh/video tự thu thập của nhóm
        own = collect_custom(Path(raw["own"]), "prefix", "own", "own")
        print(f"Dữ liệu tự thu thập: {len(own)} ảnh")
        samples += own
    if not samples:
        raise SystemExit("Không tìm thấy dữ liệu trong data/raw/. Xem README_data.md để đặt dữ liệu đúng chỗ.")
    if args.limit:
        random.Random(0).shuffle(samples)
        samples = samples[: args.limit]

    sp = cfg["split"]
    if sp.get("mode", "random") == "keep" and any(x.preset for x in samples):
        assign = {i: x.preset for i, x in enumerate(samples) if x.preset}   # giữ nguyên cách chia của bản export
        rest = [i for i, x in enumerate(samples) if not x.preset]           # GTSRB / dữ liệu tự thu thập: chia theo nhóm
        if rest:
            sub = split_groups([samples[i] for i in rest], sp, sp["seed"], True)
            assign.update({rest[j]: v for j, v in sub.items()})
        print(f"Chế độ chia: GIỮ NGUYÊN tập gốc cho {len(samples) - len(rest)} ảnh, chia theo nhóm cho {len(rest)} ảnh còn lại")
    else:
        assign = split_groups(samples, sp, sp["seed"], sp.get("group_by_track", True))
        print("Chế độ chia: NGẪU NHIÊN theo nhóm, phân tầng theo lớp")
    if sp.get("custom_group_by") == "roboflow":
        # ảnh Roboflow đã được tăng cường sẵn: val/test chỉ giữ 1 ảnh mỗi nhóm để đánh giá trên ảnh "sạch"
        seen = set()
        for i, smp in enumerate(samples):
            if smp.source == "vietnam" and assign[i] != "train":
                if smp.group in seen:
                    assign[i] = None
                else:
                    seen.add(smp.group)

    for name in ("train", "val", "test"):
        (out_root / name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / name / "labels").mkdir(parents=True, exist_ok=True)

    manifest, stats = [], {"split": defaultdict(int), "per_class": defaultdict(lambda: defaultdict(int)),
                           "source": defaultdict(lambda: defaultdict(int))}
    for i, s in enumerate(samples):
        img = cv2.imread(str(s.img_path))
        if img is None:
            print(f"[CẢNH BÁO] Không đọc được {s.img_path}")
            continue
        if not args.no_preprocess:
            img = preprocess_image(img, cfg.get("preprocess"))
        split = assign[i]
        if split is None:
            continue
        stem = s.uid
        cv2.imwrite(str(out_root / split / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(out_root / split / "labels" / f"{stem}.txt", "w") as f:
            for cid, xc, yc, bw, bh in s.boxes:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        manifest.append((f"{stem}.jpg", split, s.source, s.stratum, s.group))
        stats["split"][split] += 1
        stats["per_class"][split][str(s.stratum)] += 1
        stats["source"][s.source][split] += 1

    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "source", "class_id", "group"])
        w.writerows(manifest)
    json.dump(stats, open(out_root / "stats.json", "w"), indent=2, ensure_ascii=False)
    total = sum(stats["split"].values())
    print(f"Xong: {total} ảnh -> {dict(stats['split'])}  (thư mục: {out_root})")


if __name__ == "__main__":
    main()
