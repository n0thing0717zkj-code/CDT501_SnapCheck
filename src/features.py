"""图像质量特征（不依赖 OCR 结果，属于“无参考”特征）。"""
import cv2
import numpy as np

FEATURES = [
    "lap_var", "tenengrad", "hf_ratio",
    "bright_frac", "glare_blob_frac", "glare_contrast",
    "illum_gradient", "tilt_deg", "text_height",
    "noise_sigma", "blockiness", "contrast", "mean", "std",
]


def _order_quad(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _page_angle(gray):
    """用 A4 长宽比估计透视倾角：纸面被压扁多少，就是倾斜了多少。"""
    h, w = gray.shape[:2]
    scale = 900.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale)
    edges = cv2.Canny(small, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area_ratio = cv2.contourArea(approx) / small.size
        if not (0.15 < area_ratio < 0.95):
            continue
        q = _order_quad(approx) / scale
        ww = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        hh = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        if ww <= 1 or hh <= 1:
            continue
        aspect = hh / ww
        expected = 297.0 / 210.0
        c_ratio = aspect / expected if aspect < expected else expected / aspect
        c_ratio = float(np.clip(c_ratio, 0.0, 1.0))
        return float(np.degrees(np.arccos(c_ratio)))
    return 0.0


def extract(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32) / 255.0
    h, w = g.shape

    lap = cv2.Laplacian(g, cv2.CV_32F)
    lap_var = float(lap.var())

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))

    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    hf = g - blur
    hf_ratio = float(np.sum(hf * hf) / (np.sum(g * g) + 1e-9))

    bright = (gray >= 250).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bright_frac = float(bright.mean())
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    blob = 0.0
    glare_contrast = 1.0
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[idx, cv2.CC_STAT_AREA]
        blob = float(area / (h * w))
        if area > 0.005 * h * w:
            x, y, bw, bh = stats[idx, :4]
            patch = gray[y:y + bh, x:x + bw]
            glare_contrast = float(patch.std() / (gray.std() + 1e-6))

    # 四角亮度差，近似“阴影/光照不均”
    k = max(8, min(h, w) // 10)
    corners = [gray[:k, :k].mean(), gray[:k, -k:].mean(),
               gray[-k:, :k].mean(), gray[-k:, -k:].mean()]
    illum_gradient = float((max(corners) - min(corners)) / 255.0)

    # 文字高度：中等面积的暗色连通域
    dark = (gray < 110).astype(np.uint8)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    heights = []
    for i in range(1, n2):
        area = stats2[i, cv2.CC_STAT_AREA]
        if 25 < area < 6000:
            heights.append(stats2[i, cv2.CC_STAT_HEIGHT])
    text_height = float(np.median(heights)) if heights else 0.0

    med = cv2.medianBlur(gray, 5)
    resid = gray.astype(np.float32) - med.astype(np.float32)
    gx0 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy0 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    flat = (np.abs(gx0) + np.abs(gy0)) < 12
    noise_sigma = float(resid[flat].std()) if flat.sum() > 100 else float(resid.std())

    # JPEG 块效应：8 像素网格边界处的跳变
    d_h = np.abs(np.diff(gray.astype(np.float32), axis=1))
    grid_cols = d_h[:, 7::8]
    blockiness = float(grid_cols.mean() - d_h.mean()) if grid_cols.size else 0.0

    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float((p95 - p5) / 255.0)

    return {
        "lap_var": lap_var,
        "tenengrad": tenengrad,
        "hf_ratio": hf_ratio,
        "bright_frac": bright_frac,
        "glare_blob_frac": blob,
        "glare_contrast": glare_contrast,
        "illum_gradient": illum_gradient,
        "tilt_deg": _page_angle(gray),
        "text_height": text_height,
        "noise_sigma": noise_sigma,
        "blockiness": blockiness,
        "contrast": contrast,
        "mean": float(gray.mean() / 255.0),
        "std": float(gray.std() / 255.0),
    }


def to_vector(feat):
    return np.array([feat[k] for k in FEATURES], dtype=np.float64)
