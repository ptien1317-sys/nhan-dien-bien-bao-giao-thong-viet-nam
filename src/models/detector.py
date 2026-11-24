from ultralytics import YOLO # YOLO: Lớp của thư viện Ultralytics, hỗ trợ tải mô hình YOLO, huấn luyện, đánh giá và dự đoán.
import os   # Thư viện Python dùng để làm việc với hệ điều hành và đường dẫn tệp.

class TrafficSignDetectorModel:
    """
    Quản lý việc khởi tạo mô hình YOLOv8n/v11n cho hệ thống ITS (Mục 2.2.2).
    Sử dụng kỹ thuật Transfer Learning từ trọng số tiền huấn luyện (Pretrained weights).
    """

    # Transfer Learning sử dụng mô hình đã được huấn luyện trước trên một tập dữ liệu lớn để cải thiện hiệu suất trên một tác vụ mới
    #  có thể kết hợp với mô hình huấn luyện hiện tại của nhóm.

    def __init__(self, model_path_or_name: str = "yolov8n.pt"):
        self.model_path = model_path_or_name
        self.model = YOLO(self.model_path)

    def get_model(self) -> YOLO:
        return self.model
