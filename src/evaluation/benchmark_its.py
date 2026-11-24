import time
import numpy as np
import onnxruntime as ort

def benchmark_onnx_speed(onnx_path: str = "outputs/checkpoints/best.onnx", img_size: int = 640, num_runs: int = 100):
    """
    Đo kiểm tốc độ xử lý (FPS) và độ trễ (Latency ms) của mô hình trên thiết bị ITS
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # Khởi động làm nóng bộ nhớ (Warm-up)
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # Đo thời gian thực tế
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Đổi sang mili-giây (ms)

    avg_latency_ms = float(np.mean(latencies))
    fps = 1000.0 / avg_latency_ms
    print(f"\n=== ĐÁNH GIÁ HIỆU NĂNG HỆ THỐNG ITS ===")
    print(f" - Độ trễ trung bình (Latency): {avg_latency_ms:.2f} ms / khung hình")
    print(f" - Tốc độ xử lý (Throughput):  {fps:.2f} FPS")
    print(f" - Đánh giá chuẩn thời gian thực (>= 25 FPS): {'ĐẠT' if fps >= 25 else 'CẦN TỐI ƯU THÊM'}")
    return {"latency_ms": round(avg_latency_ms, 2), "fps": round(fps, 2)}
