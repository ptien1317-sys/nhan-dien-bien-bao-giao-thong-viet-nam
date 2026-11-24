import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import tempfile
import numpy as np
import streamlit as st
from src.inference.predictor import TrafficSignPredictor

st.set_page_config(page_title="ITS Traffic Sign Recognition Dashboard", layout="wide")

st.title("Hệ Thống Nhận Diện Biển Báo Giao Thông Thông Minh (ITS - ADAS Dashboard)")
st.caption("Mô phỏng luồng hoạt động Camera trên phương tiện & Cảnh báo thời gian thực (Mục 2.4.3)")

st.sidebar.header("Cấu hình Hệ thống ITS")
weights_input = st.sidebar.text_input("Đường dẫn trọng số (.pt):", "outputs/checkpoints/best.pt")
conf_thres = st.sidebar.slider("Ngưỡng độ tin cậy (Confidence):", 0.10, 0.95, 0.25, 0.05)
use_clahe = st.sidebar.checkbox("Bật cân bằng sáng ngoài trời", value=True)
flip_webcam = st.sidebar.checkbox("Chống lật gương Webcam (Mirror Flip)", value=True)
source_type = st.sidebar.radio("Chọn nguồn thu hình ảnh:", ["Ảnh tĩnh (Image)", "Video hành trình (MP4/AVI)", "Webcam trực tiếp"])

@st.cache_resource
def get_cached_model(weights_path):
    return TrafficSignPredictor(weights_path=weights_path, conf_thres=0.25)

predictor = get_cached_model(weights_input)
predictor.conf_thres = conf_thres

col_video, col_panel = st.columns([2, 1])

with col_panel:
    st.subheader("Bảng Cảnh Báo Người Lái (HMI)")
    kpi1, kpi2 = st.columns(2)
    fps_metric = kpi1.empty()
    lat_metric = kpi2.empty()
    alert_box = st.empty()
    st.markdown("**Nhật ký bản tin gửi về Trung tâm ITS:**")
    log_box = st.empty()

with col_video:
    st.subheader("Luồng Camera Giám Sát / Hành Trình")
    frame_window = st.empty()

    if source_type == "Ảnh tĩnh (Image)":
        uploaded_img = st.file_uploader("Tải lên ảnh đường phố / biển báo (.jpg, .png)", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            out_img, detections, fps, lat = predictor.predict_frame(bgr_img, use_clahe=use_clahe, is_static_image=True)
            frame_window.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            fps_metric.metric("Tốc độ xử lý (FPS)", f"{fps}")
            lat_metric.metric("Độ trễ (Latency)", f"{lat} ms")
            if detections:
                d0 = detections[0]
                alert_box.error(f"🚨 **{d0['alert_msg']}**\n\n- **Tên biển báo:** {d0['name']} (`{d0['code']}`)\n- **Nhóm:** {d0['group']}\n- **Độ tin cậy:** {d0['confidence']}%")
                log_box.dataframe(detections, use_container_width=True)
            else:
                alert_box.info("Chưa phát hiện biển báo nào trong khung hình.")

    elif source_type == "Video hành trình (MP4/AVI)":
        uploaded_vid = st.file_uploader("Tải lên video camera hành trình (.mp4, .avi)", type=["mp4", "avi", "mov"])
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_vid.read())
            cap = cv2.VideoCapture(tfile.name)
            history_logs = []
            stop_btn = st.button("Dừng Video")
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break
                out_img, detections, fps, lat = predictor.predict_frame(frame, use_clahe=use_clahe, is_static_image=False)
                frame_window.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                fps_metric.metric("Tốc độ xử lý (FPS)", f"{fps}")
                lat_metric.metric("Độ trễ (Latency)", f"{lat} ms")
                for d in detections:
                    if d["is_new_alert"]:
                        alert_box.error(f"🚨 **{d['alert_msg']}** — *{d['name']}* (`{d['code']}`)")
                        history_logs.insert(0, d)
                if history_logs:
                    log_box.dataframe(history_logs[:10], use_container_width=True)
            cap.release()

    elif source_type == "Webcam trực tiếp":
        run_cam = st.checkbox("Bật Camera trực tiếp")
        if run_cam:
            cap = cv2.VideoCapture(0)
            history_logs = []
            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Không mở được Webcam trên máy tính.")
                    break
                if flip_webcam:
                    frame = cv2.flip(frame, 1)
                out_img, detections, fps, lat = predictor.predict_frame(frame, use_clahe=use_clahe, is_static_image=False)
                frame_window.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                fps_metric.metric("Tốc độ xử lý (FPS)", f"{fps}")
                lat_metric.metric("Độ trễ (Latency)", f"{lat} ms")
                for d in detections:
                    if d["is_new_alert"]:
                        alert_box.error(f"🚨 **{d['alert_msg']}** — *{d['name']}* (`{d['code']}`)")
                        history_logs.insert(0, d)
                if history_logs:
                    log_box.dataframe(history_logs[:10], use_container_width=True)
            cap.release()
