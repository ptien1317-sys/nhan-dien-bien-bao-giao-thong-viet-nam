from ultralytics import YOLO
import shutil
import os

def export_model_for_its(pt_weights_path: str, output_dir: str = "outputs/checkpoints", img_size: int = 640) -> str:
    """
    Chuyển đổi mô hình từ PyTorch (.pt) sang chuẩn ONNX (.onnx) (Mục 2.2.3 & Nhánh C++).
    Giúp tăng tốc độ suy luận trên thiết bị biên ITS và đọc trực tiếp được trong C++ (OpenCV DNN / ONNX Runtime).
    """
    if not os.path.exists(pt_weights_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {pt_weights_path}")

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(pt_weights_path)

    # Xuất sang định dạng ONNX với opset=12 (tương thích hoàn hảo với cả OpenCV C++ và ONNX Runtime)
    exported_path = model.export(format="onnx", imgsz=img_size, opset=12, simplify=True)

    target_onnx_path = os.path.join(output_dir, "best.onnx")
    if os.path.abspath(str(exported_path)) != os.path.abspath(target_onnx_path):
        shutil.copy2(str(exported_path), target_onnx_path)

    print(f"[ITS Exporter] Đã xuất mô hình ONNX thành công tại: {target_onnx_path}")
    return target_onnx_path
