# cpp_inference/ - Nhanh suy luan C++ toc do cao (Thanh vien 1)

Doc `best.onnx` bang OpenCV DNN, xu ly video, phat canh bao khi Confidence >= 0.6.

## Yeu cau
- g++/clang++ ho tro C++17, CMake >= 3.12, OpenCV >= 4.5 (dnn, imgproc, videoio, highgui)
- Ubuntu: `sudo apt install build-essential cmake libopencv-dev`

## Bien dich (chay tu thu muc goc traffic_sign_its/)
```bash
cmake -S cpp_inference -B cpp_inference/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_inference/build -j
```
Tao `cpp_inference/build/bin/its_detector` (ung dung) va `its_benchmark` (do hieu nang).

## Chuan bi
```bash
python -m src.utils.export_labels --dataset configs/dataset.yaml   # tao outputs/checkpoints/labels.txt
# dat best.onnx vao outputs/checkpoints/ (Thanh vien 3 ban giao)
# chinh configs/its_runtime.yaml (source.input = video/camera)
```

## Chay
```bash
./cpp_inference/build/bin/its_detector  --source data/raw/demo.mp4 --output outputs/demo_videos/cpp_demo.mp4
./cpp_inference/build/bin/its_benchmark --source data/raw/demo.mp4          # -> outputs/logs/benchmark_cpp.json
```

## Do va so sanh voi Python (cung video, cung may)
```bash
python scripts/benchmark_python.py --backend pytorch
python scripts/benchmark_python.py --backend onnxruntime
python scripts/benchmark_python.py --backend opencv
python scripts/compare_benchmarks.py     # -> outputs/logs/benchmark_summary.md + outputs/figures/*.png
```
Dung GPU: dat `backend.target: "cuda"` trong its_runtime.yaml (can OpenCV build voi CUDA), them `--tag cpp-opencv-cuda --out outputs/logs/benchmark_cpp_cuda.json`.

## Luu y
- Dinh dang dau ra ONNX gia dinh theo Ultralytics YOLOv8/11: `[1, 4+nc, N]`. Neu Thanh vien 3 dung mo hinh khac, chi can sua `Detector::postprocess` (detector.cpp) va `decode_yolo_output` (src/utils/vision.py).
- Ten lop chi dung ASCII khi ve len hinh (cv::putText khong ho tro tieng Viet co dau).
