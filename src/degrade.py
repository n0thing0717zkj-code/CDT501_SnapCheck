"""退化库：把干净文档变成“拍坏了”的照片。

每个退化都返回 (图像, 元信息)，元信息里记录主因，作为归因的真值。
"""
import cv2
import numpy as np

FACTORS = ["blur", "glare", "tilt", "shadow", "noise"]


def _motion_kernel(length, angle_deg):
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k /= max(k.sum(), 1e-9)
    return k


def blur(img, s, rng):
    s = float(np.clip(s, 0, 1))
    if rng.random() < 0.5:
        sigma = 0.3 + 3.5 * s
        out = cv2.GaussianBlur(img, (0, 0), sigma)
        sub = "gaussian"
    else:
        length = 3 + 18 * s
        k = _motion_kernel(length, rng.uniform(-45, 45))
        out = cv2.filter2D(img, -1, k)
        sub = "motion"
    return out, {"factor": "blur", "severity": s, "subtype": sub}


def glare(img, s, rng):
    s = float(np.clip(s, 0, 1))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cx = int(rng.uniform(0.15, 0.85) * w)
    cy = int(rng.uniform(0.15, 0.85) * h)
    rx = int(w * (0.04 + 0.16 * s))
    ry = int(h * (0.03 + 0.12 * s))
    cv2.ellipse(mask, (cx, cy), (rx, ry), rng.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(6, rx * 0.35))
    mask = np.clip(mask, 0, 1)[..., None]
    strength = 0.75 + 0.25 * s
    out = img.astype(np.float32) * (1 - strength * mask) + 255.0 * (strength * mask)
    # 反光区内部对比度下降（细节被“洗掉”）
    local = cv2.GaussianBlur(out.astype(np.uint8), (0, 0), max(2, 3 * s)).astype(np.float32)
    out = out * (1 - 0.45 * mask) + local * (0.45 * mask)
    return np.clip(out, 0, 255).astype(np.uint8), {"factor": "glare", "severity": s, "subtype": "ellipse"}


def tilt(img, s, rng):
    s = float(np.clip(s, 0, 1))
    h, w = img.shape[:2]
    ang = np.radians(3 + 18 * s)
    d = np.tan(ang) * min(w, h) * 0.5
    pad = int(d) + 24
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if rng.random() < 0.5:
        # 绕水平轴：上下边被压缩
        dst = np.float32([[pad, pad + d], [pad + w, pad - d],
                          [pad + w, pad + h - d], [pad, pad + h + d]])
    else:
        # 绕竖直轴：左右边被压缩
        dst = np.float32([[pad + d, pad], [pad + w - d, pad],
                          [pad + w + d, pad + h], [pad - d, pad + h]])
    M = cv2.getPerspectiveTransform(src, dst)
    size = (w + 2 * pad, h + 2 * pad)
    out = cv2.warpPerspective(img, M, size, borderValue=(90, 90, 90))
    return out, {"factor": "tilt", "severity": s, "subtype": "perspective"}


def shadow(img, s, rng):
    s = float(np.clip(s, 0, 1))
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    proj = (np.cos(a) * xx / w + np.sin(a) * yy / h)
    proj = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-6)
    factor = 1.0 - (0.15 + 0.35 * s) * proj
    out = img.astype(np.float32) * factor[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), {"factor": "shadow", "severity": s, "subtype": "gradient"}


def noise(img, s, rng):
    s = float(np.clip(s, 0, 1))
    sigma = 2.0 + 12.0 * s
    g = np.random.default_rng(rng.randrange(1 << 30))
    out = img.astype(np.float32) + g.normal(0, sigma, img.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    quality = int(92 - 35 * s)
    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out, {"factor": "noise", "severity": s, "subtype": f"sigma{sigma:.0f}_q{quality}"}


OPS = {"blur": blur, "glare": glare, "tilt": tilt, "shadow": shadow, "noise": noise}


def apply(img, factor, severity, rng):
    return OPS[factor](img, severity, rng)


def compose(img, primary, p_sev, secondary, s_sev, rng):
    """一个主因 + 一个弱次因，标签仍记主因。"""
    out, meta = apply(img, primary, p_sev, rng)
    if secondary and secondary != primary and s_sev > 0:
        out, _ = apply(out, secondary, s_sev, rng)
        meta["secondary"] = secondary
        meta["secondary_severity"] = float(s_sev)
    return out, meta


def capture_effects(img, rng, strength=1.0):
    """模拟“手机拍出来”的样子：背景、光照不均、白平衡、噪声、轻微模糊、JPEG。

    这一步是让合成数据和实拍对齐，不是增加难度。
    """
    h, w = img.shape[:2]

    # 1) 桌面背景，页边留 2%–6%
    mx = int(w * rng.uniform(0.02, 0.06))
    my = int(h * rng.uniform(0.02, 0.06))
    bg = int(rng.choice([70, 95, 120, 150, 175, 200]))
    canvas = np.full((h + 2 * my, w + 2 * mx, 3), bg, np.uint8)
    gr = np.random.default_rng(rng.randrange(1 << 30))
    canvas = np.clip(canvas.astype(np.float32) + gr.normal(0, 2.5, canvas.shape), 0, 255).astype(np.uint8)
    canvas[my:my + h, mx:mx + w] = img
    img = canvas

    # 2) 不均匀光照
    hh, ww = img.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    g = np.cos(ang) * xx / ww + np.sin(ang) * yy / hh
    g = (g - g.min()) / max(g.max() - g.min(), 1e-6)
    factor = 1.0 - (0.08 + 0.14 * strength) * g
    img = img.astype(np.float32) * factor[..., None]

    # 3) 白平衡偏移
    gains = np.array([rng.uniform(0.97, 1.03),
                      rng.uniform(0.98, 1.02),
                      rng.uniform(0.97, 1.03)], np.float32)
    img = np.clip(img * gains, 0, 255).astype(np.uint8)

    # 4) 轻微失焦（手机对焦不完美）
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.9))

    # 5) 传感器噪声
    g2 = np.random.default_rng(rng.randrange(1 << 30))
    img = np.clip(img.astype(np.float32) + g2.normal(0, rng.uniform(1.5, 4.0), img.shape),
                  0, 255).astype(np.uint8)

    # 6) JPEG 压缩
    q = int(rng.uniform(78, 92))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 7) 长边统一到 1754，和实拍预处理一致
    s = 1754.0 / max(img.shape[:2])
    if abs(s - 1.0) > 0.01:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    return img
