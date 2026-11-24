# Hệ Thống Nhận Diện Biển Báo Giao Thông Thông Minh (ITS – ADAS)

Dự án xây dựng hệ thống **Phát hiện và Nhận diện 73 loại biển báo giao thông đường bộ Việt Nam** (theo Quy chuẩn kỹ thuật quốc gia **QCVN 41:2019/BGTVT**) tích hợp vào kiến trúc **Hệ thống giao thông thông minh (ITS)** và **Hỗ trợ người lái nâng cao (ADAS)**.

Hệ thống kết hợp quy trình tiền xử lý ảnh ngoài trời (**CLAHE + Lọc song phương Bilateral**), tăng cường dữ liệu mô phỏng 6 điều kiện thời tiết/camera, huấn luyện mô hình học sâu gọn nhẹ (**YOLOv8n**), chuẩn hóa trọng số sang định dạng trung gian **ONNX (`opset 18`)**, nhánh thực thi tốc độ cao bằng **C++ (OpenCV DNN)** và giao diện **Dashboard Python (Streamlit)** cảnh báo tiếng Việt thời gian thực.

## 1. Kết Quả Thực Nghiệm Chính

### 1.1. Độ chính xác mô hình (73 lớp biển báo Việt Nam)
* **Bộ dữ liệu:** `Vietnamese Traffic Signs – have augment exposure` (Roboflow Universe, 7.987 ảnh ban đầu / 18.997 khung nhãn; sau tăng cường tập Train đạt tổng 15.083 ảnh).
* **Mô hình:** `YOLOv8n` (`640×640`, trọng số `best.pt` ~6,3 MB, `best.onnx` ~12,8 MB).

| Kịch bản huấn luyện | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| YOLOv8n (Đỉnh hội tụ `best.pt` – Epoch 49) | **86,54%** | **82,37%** | **84,40%** | **89,06%** | **72,93%** |
| YOLOv8n (Kết thúc huấn luyện – Epoch 50) | 85,51% | 82,70% | 84,08% | 88,60% | 72,81% |

### 1.2. Hiệu năng tốc độ & Độ trễ cảnh báo (đo thuần CPU)
Đo trên cùng tệp video `data/raw/demo.mp4` (`1280×720`, 30 FPS; 20 khung warm-up + 100 khung đo) trên Windows 11, CPU Intel 12 luồng, Python 3.12.6, OpenCV 5.0.0, ONNX Runtime 1.31.1:

| Môi trường thực thi (CPU) | FPS (pipeline) | FPS (end-to-end) | Tiền xử lý / Suy luận / NMS (ms) | Độ trễ TB / p95 / p99 (ms) | Ngưỡng < 100 ms |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Python – PyTorch (`best.pt`)** | **15,3** | **14,6** | 1,77 / 62,78 / 0,73 | **65,96 / 79,72 / 94,87** | **Đạt (cả p99)** |
| Python – OpenCV DNN (`best.onnx`) | 10,0 | 9,8 | 7,52 / 90,13 / 2,05 | 99,71 / 166,61 / 207,29 | Đạt mức TB |
| Python – ONNX Runtime (`best.onnx`) | 5,5 | 5,4 | 12,59 / 165,74 / 3,13 | 181,52 / 366,99 / 430,65 | Chưa đạt |

## 2. Cấu Trúc Thư Mục Dự Án (`traffic_sign_its_Group_v3/`)

```text
traffic_sign_its_Group_v3/
├── .gitignore                         # Loại trừ dữ liệu ảnh lớn và môi trường ảo khi đẩy Git
├── README.md                          # Tài liệu hướng dẫn tổng quan dự án
├── requirements.txt                   # Danh sách thư viện Python cần thiết
├── run_all.py                         # Chạy trọn bộ pipeline xử lý & tăng cường dữ liệu
├── app/
│   ├── assets/                        # Thư mục lưu tài nguyên giao diện (icon/âm thanh mở rộng)
│   └── dashboard.py                   # Giao diện ADAS Dashboard (Streamlit: Ảnh, Video, Webcam)
├── configs/
│   ├── dataset.yaml                   # Cấu hình đường dẫn dữ liệu & danh sách 73 lớp biển báo VN
│   ├── its_runtime.yaml               # Tham số runtime dùng chung cho nhánh C++ và Python
│   └── train_config.yaml              # Siêu tham số huấn luyện (epochs, batch, lr, fliplr=0.0)
├── cpp_inference/                     # Nhánh suy luận tốc độ cao bằng C++ (OpenCV DNN)
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── include/                       # alert.hpp, detector.hpp, its_config.hpp, its_utils.hpp
│   └── src/                           # alert.cpp, benchmark.cpp, detector.cpp, its_config.cpp, main.cpp
├── data/
│   ├── README_data.md                 # Hướng dẫn tải & chuẩn bị bộ dữ liệu chi tiết
│   ├── raw/                           # Dữ liệu thô (có sẵn video mẫu demo.mp4)
│   └── processed/                     # (Trống sẵn — tự động sinh ra khi chạy python run_all.py)
├── notebooks/
│   └── train_yolov8.ipynb             # Sổ tay huấn luyện YOLOv8n trên Google Colab GPU
├── outputs/
│   ├── checkpoints/                   # best.pt, best.onnx, last.pt, labels.txt (73 lớp)
│   ├── demo_videos/                   # (Tự động lưu video kết quả khi chạy scripts/inference_video.py)
│   ├── figures/                       # Biểu đồ PR/F1/Confusion Matrix, pipeline_its.png, fps_comparison.png, latency_breakdown.png
│   └── logs/                          # benchmark_py_*.json, benchmark_summary.{csv,md} & traffic_sign/ (results.csv, args.yaml)
├── scripts/
│   ├── benchmark_python.py            # Đo FPS & độ trễ cho 3 backend (pytorch, onnxruntime, opencv)
│   ├── compare_benchmarks.py          # Gộp các tệp benchmark_*.json thành bảng & 2 biểu đồ PNG
│   ├── evaluate.py                    # Xuất bảng độ chính xác (mAP, F1) & đo tốc độ ONNX
│   ├── inference_video.py             # Nhận diện video và xuất MP4 vào outputs/demo_videos/
│   └── train.py                       # Khởi chạy huấn luyện mô hình
├── src/
│   ├── __init__.py
│   ├── data_pipeline/                 # import_vn_dataset.py, preprocess.py, augmentation.py, prepare_dataset.py, augment_dataset.py, class_report.py, collect_own.py, visualize.py
│   ├── evaluation/                    # metrics.py, benchmark_its.py
│   ├── inference/                     # predictor.py, alert_manager.py
│   ├── models/                        # detector.py, exporter.py
│   ├── training/                      # train.py
│   └── utils/                         # config.py, export_labels.py, logger.py, timer.py, vision.py
└── tests/                             # Thư mục kiểm thử đơn vị (mở rộng)
```

> **Lưu ý về các thư mục trống:**
> * `data/processed/` không đóng gói sẵn trong kho mã nguồn do dung lượng bộ ảnh lớn (~15.083 ảnh). Để tái tạo, làm theo `data/README_data.md` rồi chạy `python run_all.py`.
> * `outputs/demo_videos/` sẽ tự động được ghi tệp kết quả (`demo_result.mp4`) khi chạy lệnh `scripts/inference_video.py` bên dưới.

## 3. Hướng Dẫn Cài Đặt & Chạy Nhanh

### Bước 1: Tạo môi trường ảo và cài đặt thư viện
```bash
python -m venv venv
source venv/Scripts/activate          # Windows (Git Bash)
# Hoặc PowerShell: .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Bước 2: Chạy giao diện mô phỏng ADAS Dashboard (Ảnh tĩnh, Video, Webcam)
```bash
streamlit run app/dashboard.py
```
Giao diện tự động bật chuỗi tiền xử lý ngoài trời (**CLAHE + lọc song phương**), hỗ trợ tải ảnh biển báo, video camera hành trình hoặc bật Webcam trực tiếp (kèm tùy chọn chống lật gương *Mirror Flip*).

### Bước 3: Xem báo cáo độ chính xác & Xuất video nhận diện vào `outputs/demo_videos/`
```bash
# In bảng chỉ số Precision, Recall, F1-Score, mAP và đo tốc độ ONNX
python scripts/evaluate.py

# Xử lý video hành trình và lưu kết quả vào outputs/demo_videos/demo_result.mp4
python scripts/inference_video.py --source data/raw/demo.mp4 --out outputs/demo_videos/demo_result.mp4
```

### Bước 4: Đo kiểm tốc độ (Benchmark) và vẽ biểu đồ so sánh
```bash
python scripts/benchmark_python.py --backend pytorch --source data/raw/demo.mp4 --warmup 20 --frames 100
python scripts/benchmark_python.py --backend onnxruntime --source data/raw/demo.mp4 --warmup 20 --frames 100
python scripts/benchmark_python.py --backend opencv --source data/raw/demo.mp4 --warmup 20 --frames 100
python scripts/compare_benchmarks.py
```

### Bước 5: Biên dịch và chạy nhánh C++ tốc độ cao (yêu cầu CMake ≥ 3.12, OpenCV ≥ 4.11)
```bash
python -m src.utils.export_labels --dataset configs/dataset.yaml --out outputs/checkpoints/labels.txt
cmake -S cpp_inference -B cpp_inference/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_inference/build -j

./cpp_inference/build/bin/its_detector --source data/raw/demo.mp4
./cpp_inference/build/bin/its_benchmark --source data/raw/demo.mp4
```

## 4. Yêu Cầu Tương Thích
* **Python:** 3.10 – 3.12 (`ultralytics`, `opencv-python`, `onnxruntime`, `streamlit`, `albumentations`, `pyyaml`, `pandas`, `matplotlib`).
* **OpenCV cho `best.onnx` (ONNX opset 18):** Yêu cầu OpenCV phiên bản mới (đã kiểm chứng hoạt động ổn định trên OpenCV 4.11 và 5.0.0; các bản cũ như 4.6 không nạp được opset 18).
