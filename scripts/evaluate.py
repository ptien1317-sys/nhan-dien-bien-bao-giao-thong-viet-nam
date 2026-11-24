import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.benchmark_its import benchmark_onnx_speed
import pandas as pd

if __name__ == "__main__":
    csv_path = "outputs/logs/traffic_sign/results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        last_row = df.iloc[-1]
        p = float(last_row.get("metrics/precision(B)", 0))
        r = float(last_row.get("metrics/recall(B)", 0))
        map50 = float(last_row.get("metrics/mAP50(B)", 0))
        map95 = float(last_row.get("metrics/mAP50-95(B)", 0))
        f1 = 2 * p * r / (p + r + 1e-8)

        print("\n=======================================================")
        print("   BÁO CÁO ĐỘ CHÍNH XÁC MÔ HÌNH AI                      ")
        print("=======================================================")
        print(f" - Precision (Độ chuẩn xác):     {p*100:.2f}%")
        print(f" - Recall (Độ phủ):              {r*100:.2f}%")
        print(f" - F1-Score:                     {f1*100:.2f}%")
        print(f" - mAP@0.5:                      {map50*100:.2f}%")
        print(f" - mAP@0.5:0.95:                 {map95*100:.2f}%")

    if os.path.exists("outputs/checkpoints/best.onnx"):
        benchmark_onnx_speed("outputs/checkpoints/best.onnx")
