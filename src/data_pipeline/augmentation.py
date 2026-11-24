"""Tăng cường dữ liệu mô phỏng các thách thức của ITS: mưa, sương mù, nhòe chuyển động,
thiếu sáng, chói đèn pha, nhiễu cảm biến.

Tất cả phép biến đổi đều là biến đổi MÀU/ĐỘ SÁNG (không dịch chuyển hình học) nên
nhãn bounding box (YOLO) giữ nguyên, không cần tính lại.
"""
from __future__ import annotations

import cv2
import numpy as np


def add_rain(img, rng, intensity=None, angle=None):
    h, w = img.shape[:2]
    intensity = rng.uniform(0.4, 1.0) if intensity is None else intensity
    angle = rng.uniform(-20, 20) if angle is None else angle
    n = int(intensity * (h * w) / 90)
    layer = np.zeros((h, w), np.uint8)
    length = max(6, int(0.12 * h))
    dx = int(length * np.sin(np.radians(angle)))
    dy = int(length * np.cos(np.radians(angle)))
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x + dx), int(y + dy)), int(rng.integers(150, 255)), 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    streak = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = img.astype(np.float32) * (1 - 0.15 * intensity)          # trời mưa: tối nhẹ
    out = out + 0.55 * streak
    return np.clip(out, 0, 255).astype(np.uint8)


def add_fog(img, rng, density=None):
    h, w = img.shape[:2]
    density = rng.uniform(0.3, 0.75) if density is None else density
    # sương dày hơn ở xa (phía trên khung hình) mô phỏng chiều sâu
    grad = np.linspace(1.0, 0.55, h, dtype=np.float32)[:, None]
    noise = cv2.resize(rng.random((max(2, h // 32), max(2, w // 32))).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    alpha = np.clip(density * grad * (0.75 + 0.5 * noise), 0, 0.92)[..., None]
    fog_color = np.full(img.shape, 215, dtype=np.float32)
    out = img.astype(np.float32) * (1 - alpha) + fog_color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def add_motion_blur(img, rng, ksize=None, angle=None):
    ksize = int(rng.integers(5, 16)) if ksize is None else ksize
    angle = rng.uniform(0, 180) if angle is None else angle
    k = np.zeros((ksize, ksize), np.float32)
    k[ksize // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (ksize, ksize))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k)


def add_low_light(img, rng, gamma=None):
    gamma = rng.uniform(1.8, 3.0) if gamma is None else gamma
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    out = cv2.LUT(img, table).astype(np.float32)
    out += rng.normal(0, 4, img.shape)                              # nhiễu cảm biến khi thiếu sáng
    return np.clip(out, 0, 255).astype(np.uint8)


def add_glare(img, rng):
    """Chói đèn pha / ngược sáng: một quầng sáng tròn mềm."""
    h, w = img.shape[:2]
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    radius = rng.uniform(0.25, 0.6) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    mask = np.clip(1 - d, 0, 1) ** 2 * rng.uniform(0.6, 1.0)
    out = img.astype(np.float32) + mask[..., None] * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def add_noise(img, rng, sigma=None):
    sigma = rng.uniform(5, 20) if sigma is None else sigma
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


EFFECTS = {
    "rain": add_rain,
    "fog": add_fog,
    "motion_blur": add_motion_blur,
    "low_light": add_low_light,
    "glare": add_glare,
    "noise": add_noise,
}


def augment_image(img, rng: np.random.Generator, prob: dict | None = None, force_one: bool = True):
    """Áp dụng ngẫu nhiên một tổ hợp hiệu ứng. Trả về (ảnh, danh sách hiệu ứng đã dùng)."""
    prob = prob or {k: 0.3 for k in EFFECTS}
    applied = [name for name, p in prob.items() if name in EFFECTS and rng.random() < p]
    if not applied and force_one:
        applied = [str(rng.choice(list(EFFECTS)))]
    out = img
    for name in applied:
        out = EFFECTS[name](out, rng)
    return out, applied
