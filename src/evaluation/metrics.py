from ultralytics import YOLO

def evaluate_accuracy(weights_path: str = "outputs/checkpoints/best.pt", data_config: str = "configs/dataset.yaml"):
    """
    Đánh giá độ chính xác của mô hình trên tập Test/Val (Mục 2.4.1):
    Trích xuất mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score và Confusion Matrix.
    """
    model = YOLO(weights_path)
    metrics = model.val(data=data_config, split="test", project="outputs/figures", name="eval_report", exist_ok=True)

    p = float(metrics.box.mp)# Precision cho biết trong số các dự đoán dương tính của mô hình, tỷ lệ dự đoán đúng là bao nhiêu.
    r = float(metrics.box.mr)# Recall cho biết trong số các đối tượng thực sự có trong dữ liệu,tỷ lệ được mô hình phát hiện đúng bn.
    f1 = 2 * p * r / (p + r + 1e-8) # F1-Score là trung bình điều hòa của Precision và Recall

    summary = {
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1-Score": round(f1, 4),
        "mAP@0.5": round(float(metrics.box.map50), 4),
        "mAP@0.5:0.95": round(float(metrics.box.map), 4),
    }
    print("\n=== KẾT QUẢ ĐÁNH GIÁ ĐỘ CHÍNH XÁC (MỤC 2.4.1) ===")
    for k, v in summary.items():
        print(f" - {k}: {v}")
    return summary
