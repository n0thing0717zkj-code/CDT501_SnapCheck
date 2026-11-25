"""手机照片预处理：只转正 + 缩放，不做透视矫正。

为什么要转正：照片像素本身可能是横的/倒的，OCR 直接读会全军覆没。
为什么不做透视矫正：倾斜是我们想测量的退化，矫正掉就没意义了。
"""
import cv2
import numpy as np

from src import ocr

TARGET_LONG_SIDE = 1754      # 与训练集 A4@150dpi 一致


def _is_cjk(ch):
    return "\u4e00" <= ch <= "\u9fff"


def _small(img, long_side=1200):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def best_rotation(img, lang=None):
    """四个方向各 OCR 一次，选汉字最多（再比置信度）的方向。"""
    small = _small(img)
    best_k, best_score = 0, -1.0
    detail = []
    for k in range(4):
        cand = np.rot90(small, k)
        try:
            text, conf, _ = ocr.run(cand, lang=lang, psm=6)
        except Exception:
            text, conf = "", 0.0
        cjk = sum(1 for ch in text if _is_cjk(ch))
        score = cjk + 0.05 * float(conf)
        detail.append((k, cjk, round(float(conf), 1)))
        if score > best_score:
            best_k, best_score = k, score
    return best_k, detail


def resize_for_model(img):
    h, w = img.shape[:2]
    s = TARGET_LONG_SIDE / max(h, w)
    if abs(s - 1.0) < 0.02:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def prepare(img, lang=None):
    k, detail = best_rotation(img, lang=lang)
    if k:
        img = np.rot90(img, k)
    return resize_for_model(img), {"rot_k": int(k), "rotation_detail": detail}
