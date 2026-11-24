"""Tiền xử lý ảnh ngoài trời: cân bằng sáng (CLAHE) và lọc nhiễu."""
from __future__ import annotations

import cv2
import numpy as np


def balance_brightness(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE trên kênh L của không gian LAB: tăng tương phản cục bộ nhưng giữ nguyên màu biển báo."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Lọc nhiễu. bilateral giữ được cạnh viền biển báo; nlm chậm hơn nhưng sạch hơn."""
    method = (method or "none").lower()
    if method == "none":
        return img
    if method == "gaussian":
        return cv2.GaussianBlur(img, (3, 3), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    raise ValueError(f"Phương pháp lọc nhiễu không hợp lệ: {method}")


def limit_size(img: np.ndarray, max_side: int = 0) -> np.ndarray:
    """Thu nhỏ giữ tỷ lệ. Nhãn YOLO chuẩn hóa (0-1) vẫn đúng sau khi resize đồng đều."""
    if not max_side:
        return img
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def preprocess_image(img: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Chuỗi tiền xử lý theo mục `preprocess` trong configs/dataset.yaml."""
    cfg = cfg or {}
    out = limit_size(img, int(cfg.get("max_side", 0)))
    out = denoise(out, cfg.get("denoise", "none"))
    if cfg.get("clahe", True):
        out = balance_brightness(out, float(cfg.get("clahe_clip_limit", 2.0)),
                                 int(cfg.get("clahe_tile_grid", 8)))
    return out
