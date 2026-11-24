"""Gop cac file outputs/logs/benchmark_*.json thanh bang so sanh + bieu do cho bao cao (muc 2.4.2).

Chay:  python scripts/compare_benchmarks.py [--baseline py-pytorch]
Dau ra: outputs/logs/benchmark_summary.csv, benchmark_summary.md,
        outputs/figures/fps_comparison.png, outputs/figures/latency_breakdown.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR, FIG_DIR = Path("outputs/logs"), Path("outputs/figures")


def load_results(pattern="benchmark_*.json"):
    rows = []
    for p in sorted(LOG_DIR.glob(pattern)):
        if p.name == "benchmark_summary.json":
            continue
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None, help="tag dung lam moc tinh toc do (mac dinh: py-pytorch, neu khong co thi dong Python dau tien)")
    args = ap.parse_args()

    rows = load_results()
    if not rows:
        raise SystemExit("Chua co file outputs/logs/benchmark_*.json - hay chay cac phep do truoc.")
    base = next((r for r in rows if r["tag"] == (args.baseline or "py-pytorch")), None) \
        or next((r for r in rows if r["language"] == "Python"), rows[0])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["tag", "language", "backend", "frames", "fps_pipeline", "fps_end_to_end", "pre_ms", "inf_ms",
              "post_ms", "total_ms", "latency_mean_ms", "latency_p95_ms", "speedup_vs_baseline"]
    table = []
    for r in rows:
        table.append({
            "tag": r["tag"], "language": r["language"], "backend": r["backend"], "frames": r["frames"],
            "fps_pipeline": round(r["fps_pipeline"], 1), "fps_end_to_end": round(r["fps_end_to_end"], 1),
            "pre_ms": round(r["preprocess_ms"]["mean"], 2), "inf_ms": round(r["inference_ms"]["mean"], 2),
            "post_ms": round(r["postprocess_ms"]["mean"], 2), "total_ms": round(r["total_ms"]["mean"], 2),
            "latency_mean_ms": round(r["alert_latency_ms"]["mean"], 2),
            "latency_p95_ms": round(r["alert_latency_ms"]["p95"], 2),
            "speedup_vs_baseline": round(r["fps_pipeline"] / base["fps_pipeline"], 2),
        })
    with (LOG_DIR / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(table)

    md = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    md += ["| " + " | ".join(str(t[k]) for k in fields) + " |" for t in table]
    (LOG_DIR / "benchmark_summary.md").write_text("\n".join(md) + f"\n\nMoc so sanh: {base['tag']}\n", encoding="utf-8")

    tags = [t["tag"] for t in table]
    colors = ["#1f77b4" if t["language"] == "C++" else "#ff7f0e" for t in table]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(tags, [t["fps_pipeline"] for t in table], color=colors)
    ax.axhline(20, color="red", ls="--", lw=1, label="Muc tieu toi thieu 20 FPS")
    ax.bar_label(bars, fmt="%.1f")
    ax.set_ylabel("FPS (pipeline)"); ax.set_title("So sanh toc do xu ly giua cac moi truong")
    ax.legend(); plt.xticks(rotation=15); plt.tight_layout()
    fig.savefig(FIG_DIR / "fps_comparison.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pre = [t["pre_ms"] for t in table]; inf = [t["inf_ms"] for t in table]; post = [t["post_ms"] for t in table]
    ax.bar(tags, pre, label="Tien xu ly")
    ax.bar(tags, inf, bottom=pre, label="Suy luan (inference)")
    ax.bar(tags, post, bottom=[a + b for a, b in zip(pre, inf)], label="Hau xu ly (NMS)")
    ax.axhline(100, color="red", ls="--", lw=1, label="Nguong do tre 100 ms")
    ax.set_ylabel("ms / khung hinh"); ax.set_title("Phan ra thoi gian xu ly theo giai doan")
    ax.legend(); plt.xticks(rotation=15); plt.tight_layout()
    fig.savefig(FIG_DIR / "latency_breakdown.png", dpi=200); plt.close(fig)
    print((LOG_DIR / "benchmark_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
