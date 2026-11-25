"""归因（哪一项最糟）+ 建议 + 反事实验证。

归因用一个浅层决策树，输入是质量特征；反事实只用来回答
“按建议修完之后，预测错字率会降到多少”。
"""
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.features import extract
from src.model import predict

ATTR_FEATURES = ["tilt_deg", "glare_blob_frac", "illum_gradient",
                 "lap_var", "noise_sigma", "text_height", "contrast", "hf_ratio"]

ADVICE = {
    "blur": "保持手机稳定，轻触屏幕对焦后重拍 / hold steady, tap to focus, retake",
    "glare": "左右移动约 10 cm 避开反光，或改变光源角度 / move ~10 cm to avoid the glare",
    "tilt": "手机正对文档，尽量与纸面平行 / face the page squarely",
    "shadow": "调整光源位置，避免手或身体挡光 / move the light source or your hand",
    "noise": "增加照明，或开启夜景/降噪模式 / add light or enable night mode",
    "none": "照片可用，无需重拍 / usable, no retake needed",
}


def train_attributor(rows, max_depth=None):
    from sklearn.ensemble import RandomForestClassifier
    X = np.array([[r["features"][k] for k in ATTR_FEATURES] for r in rows])
    y = np.array([r["factor"] for r in rows])
    clf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, max_depth=max_depth,
        class_weight="balanced", random_state=0, n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def predict_factor(clf, feat):
    x = np.array([[feat[k] for k in ATTR_FEATURES]])
    proba = clf.predict_proba(x)[0]
    i = int(np.argmax(proba))
    return str(clf.classes_[i]), float(proba[i])


def _deblur(img):
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 2.2, blur, -1.2, 0)


def _deglare(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray >= 246).astype(np.uint8) * 255
    if mask.sum() == 0:
        return img
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def _order_quad(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _page_quad(gray):
    h, w = gray.shape[:2]
    scale = 900.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale)
    edges = cv2.Canny(small, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and 0.1 < cv2.contourArea(approx) / small.size < 0.95:
            return _order_quad(approx) / scale
    return None


def _detilt(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad = _page_quad(gray)
    if quad is None:
        return img
    h, w = gray.shape[:2]
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(246, 246, 246))


def _deshadow(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(small, (0, 0), 15)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = np.maximum(bg, 30).astype(np.float32)
    out = img.astype(np.float32) * (200.0 / bg[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _denoise(img):
    return cv2.bilateralFilter(img, 5, 45, 45)


CORRECT = {
    "blur": _deblur,
    "glare": _deglare,
    "tilt": _detilt,
    "shadow": _deshadow,
    "noise": _denoise,
    "none": lambda x: x,
}


def attribute(img, model, attributor=None, base_feat=None):
    base_feat = base_feat or extract(img)
    base_pred = predict(model, base_feat)

    if attributor is None:
        return {"base_pred": base_pred, "top_factor": None, "confidence": 0.0,
                "advice": ADVICE["none"], "pred_after_fix": base_pred,
                "corrected_img": img, "deltas": {}}

    factor, conf = predict_factor(attributor, base_feat)
    if conf < 0.25:
        factor, conf = "none", 0.0

    fixed = CORRECT.get(factor, lambda x: x)(img)
    after = predict(model, extract(fixed)) if factor != "none" else base_pred
    after = min(base_pred, after)

    return {
        "base_pred": base_pred,
        "top_factor": None if factor == "none" else factor,
        "confidence": round(conf, 3),
        "advice": ADVICE.get(factor, ADVICE["none"]),
        "pred_after_fix": float(after),
        "corrected_img": fixed,
        "delta": float(base_pred - after),
    }
