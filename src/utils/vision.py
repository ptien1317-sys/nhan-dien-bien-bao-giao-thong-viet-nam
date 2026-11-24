"""Tien/hau xu ly cho mo hinh YOLO xuat ONNX - logic giong het Detector (C++)."""
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize giu ti le + de vien xam. new_shape = (rong, cao).
    Tra ve (anh_vuong, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    nw_, nh_ = new_shape
    scale = min(nw_ / w, nh_ / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    pad_x, pad_y = (nw_ - nw) // 2, (nh_ - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((nh_, nw_, 3), color, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def to_blob(canvas):
    """BGR HWC uint8 -> RGB NCHW float32 [0,1]."""
    return cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True, crop=False)


def decode_yolo_output(output, scale, pad, frame_shape, conf_thr=0.25, nms_thr=0.45, max_det=100):
    """Giai ma dau ra [1, 4+nc, N] hoac [1, N, 4+nc] thanh danh sach
    (x, y, w, h, conf, class_id) tren khung hinh goc, da qua NMS theo tung lop."""
    pred = np.squeeze(np.asarray(output), axis=0)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                                   # -> N x (4+nc)
    cls_scores = pred[:, 4:]
    class_ids = cls_scores.argmax(axis=1)
    confs = cls_scores[np.arange(len(pred)), class_ids]
    keep = confs >= conf_thr
    if not keep.any():
        return []
    pred, class_ids, confs = pred[keep], class_ids[keep], confs[keep]

    fh, fw = frame_shape[:2]
    x1 = (pred[:, 0] - pred[:, 2] / 2 - pad[0]) / scale
    y1 = (pred[:, 1] - pred[:, 3] / 2 - pad[1]) / scale
    x2 = x1 + pred[:, 2] / scale
    y2 = y1 + pred[:, 3] / scale
    x1, y1 = np.clip(x1, 0, fw), np.clip(y1, 0, fh)
    x2, y2 = np.clip(x2, 0, fw), np.clip(y2, 0, fh)
    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    boxes, class_ids, confs = boxes[valid], class_ids[valid], confs[valid]
    if len(boxes) == 0:
        return []

    shifted = boxes.copy()
    shifted[:, 0] += class_ids * 8192                   # NMS tinh rieng tung lop
    idx = cv2.dnn.NMSBoxes(shifted.tolist(), confs.tolist(), conf_thr, nms_thr, top_k=max_det)
    idx = np.array(idx).flatten()
    return [(int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3]),
             float(confs[i]), int(class_ids[i])) for i in idx]


def filter_alerts(detections, alert_conf=0.6):
    """Khoi canh bao: chi giu ket qua Confidence >= nguong (2.2.3, buoc 5)."""
    return [d for d in detections if d[4] >= alert_conf]


def draw_detections(frame, detections, labels, alert_conf=0.6):
    """Ve khung bao + nhan len khung hinh (ASCII)."""
    for x, y, w, h, conf, cid in detections:
        color = (0, 200, 0) if conf >= alert_conf else (160, 160, 160)
        name = labels[cid] if 0 <= cid < len(labels) else f"class_{cid}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame
