# Hướng dẫn dữ liệu – Thành viên 2 (pipeline dữ liệu biển báo)

## Cấu trúc code
| File | Vai trò |
|---|---|
| `configs/dataset.yaml` | Đường dẫn, cách chia tập, tham số tiền xử lý / tăng cường, danh sách lớp (`nc`, `names`) |
| `src/data_pipeline/import_vn_dataset.py` | Nhập bộ dữ liệu YOLO (Roboflow, Kaggle...) vào `data/raw/vietnam/`, tự cập nhật `nc`/`names` trong `configs/dataset.yaml` |
| `src/data_pipeline/preprocess.py` | CLAHE (cân bằng sáng trên kênh L của LAB) + lọc nhiễu song phương |
| `src/data_pipeline/prepare_dataset.py` | Tiền xử lý, chia train/val/test (giữ chia gốc hoặc chia ngẫu nhiên theo nhóm), ghi `data/processed/` |
| `src/data_pipeline/augmentation.py` | Mưa, sương mù, nhòe chuyển động, thiếu sáng, chói đèn pha, nhiễu cảm biến |
| `src/data_pipeline/augment_dataset.py` | Sinh ảnh tăng cường cho tập train, ghi `augmentation_log.csv` |
| `src/data_pipeline/class_report.py` | Thống kê đúng theo từng khung nhãn, vẽ `outputs/figures/dataset_distribution.png`, ghi `outputs/class_report.csv` |
| `src/data_pipeline/visualize.py` | Vẽ hình minh họa augmentation (`visualize augment --image ...`) |
| `run_all.py` | Chạy toàn bộ pipeline bằng một lệnh và in số liệu cho báo cáo |

## Chạy từ đầu (Windows PowerShell, thư mục gốc project)
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Tải bản export YOLOv8 từ Roboflow (hoặc Kaggle), giải nén vào một thư mục có data.yaml
python -m src.data_pipeline.import_vn_dataset --src "C:\vn_signs2"

# 2. Chạy toàn bộ: tiền xử lý -> chia tập -> tăng cường -> thống kê -> hình
python run_all.py
```
Kết quả: `data/processed/{train,val,test}/{images,labels}`, `manifest.csv`, `stats.json`,
`outputs/class_report.csv`, `outputs/figures/*.png`.

## Lưu ý
* Nhãn định dạng YOLO: `class_id x_center y_center width height` (chuẩn hóa 0–1).
* `import_vn_dataset` **thay thế** dữ liệu đã nhập lần trước và **ghi đè** `nc`/`names` trong `configs/dataset.yaml`.
* Bộ Roboflow có ảnh tăng cường sẵn (tên file chứa `.rf.`): pipeline gom các bản của cùng một ảnh gốc vào cùng một tập
  và giảm tăng cường thêm xuống 1 bản/ảnh.
* Ảnh train đã qua CLAHE + lọc song phương -> khâu suy luận (Thành viên 1 C++, Thành viên 4 Python)
  phải áp dụng cùng chuỗi tiền xử lý (`preprocess_image`).
* Bàn giao cho Thành viên 3, 4: `data/processed/` + `configs/dataset.yaml` (số lớp, tên lớp).

## Kết hợp GTSRB (tùy chọn)
GTSRB dùng 43 lớp biển Đức, khác bảng lớp Việt Nam, nên **chỉ các lớp có ánh xạ** trong `configs/dataset.yaml > gtsrb_merge.map`
mới được đưa vào, và được đổi sang mã lớp Việt Nam (các lớp khác bị bỏ). Bảng ánh xạ hiện tại (đề xuất, nhóm cần xác nhận):
GTSRB 17→P-102, 9→P-125, {0,1,2,3,4,5,7,8}→P-127, 27→W-224, 28→W-225, 25→W-227, 40→R-303.
1. Tải GTSRB bản Kaggle, giải nén vào `data/raw/gtsrb/` (có `Train.csv` và thư mục `Train/`).
2. Trong `configs/dataset.yaml` đặt `gtsrb_merge.enabled: true` (`max_per_class` giới hạn số ảnh GTSRB mỗi lớp đích).
3. `python run_all.py`. Ảnh GTSRB được chia train/val/test theo từng track (không rò rỉ); ảnh Roboflow giữ nguyên tập gốc.
Lưu ý: ảnh GTSRB là ảnh cắt sát biển báo nên khác ảnh đường phố; hãy theo dõi ảnh hưởng đến độ chính xác khi huấn luyện.

## Dữ liệu tự thu thập (ảnh / video thực tế)
```powershell
python -m src.data_pipeline.collect_own classes                       # tạo data/raw/own/classes.txt cho công cụ gán nhãn
python -m src.data_pipeline.collect_own frames --video "D:\video\quay1.mp4" --name quay1   # tách khung hình (1 khung/giây)
```
Gán nhãn các khung hình bằng LabelImg (chọn định dạng **YOLO**, nạp `classes.txt`), lưu file `.txt` vào `data/raw/own/labels/`.
Ảnh cùng video luôn được xếp vào cùng một tập. Sau đó chạy `python run_all.py` để gộp vào bộ dữ liệu.
Ảnh tự thu thập cần dùng đúng mã lớp trong `configs/dataset.yaml` (`classes.txt` đã đúng thứ tự).
