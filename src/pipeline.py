"""端到端：一张图 -> 预测错字率 -> 能不能用 -> 主因 -> 建议 -> 修正后预测。"""
import cv2
import numpy as np

from src.attribute import attribute
from src.features import extract
from src.model import predict


def analyze(img, model, threshold, tau=0.02, do_attribute=True, attributor=None):
    feat = extract(img)
    pred = predict(model, feat)
    accept = bool(pred <= threshold)

    result = {
        "pred_cer": round(pred, 4),
        "threshold": round(float(threshold), 4),
        "accept": accept,
        "status": "accept" if accept else "reject",
        "features": {k: round(float(v), 4) for k, v in feat.items()},
    }

    if do_attribute:
        attr = attribute(img, model, attributor=attributor, base_feat=feat)
        result.update({
            "top_factor": attr["top_factor"],
            "factor_confidence": attr.get("confidence", 0.0),
            "advice": attr["advice"],
            "pred_after_fix": round(attr["corrected_pred"], 4) if "corrected_pred" in attr else round(attr["pred_after_fix"], 4),
            "deltas": attr.get("deltas", {}),
        })
        result["_corrected_img"] = attr["corrected_img"]

    return result


def annotate(img, result):
    """在图上画结果卡片；中文用英文占位，避免 OpenCV 字体问题。"""
    vis = img.copy()
    h, w = vis.shape[:2]
    scale = min(1.0, 1100.0 / max(h, w))
    if scale < 1.0:
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok = result["accept"]
    color = (40, 150, 40) if ok else (30, 30, 200)
    lines = [
        f"pred CER {result['pred_cer'] * 100:.1f}%   (limit {result['threshold'] * 100:.1f}%)",
        ("ACCEPT" if ok else "REJECT") + (f"   main cause: {result.get('top_factor')}" if result.get("top_factor") else ""),
    ]
    if result.get("pred_after_fix") is not None:
        lines.append(f"after fix {result['pred_after_fix'] * 100:.1f}%")

    pad, lh = 18, 34
    box_w = 640
    cv2.rectangle(vis, (12, 12), (12 + box_w, 12 + pad * 2 + lh * len(lines)), (255, 255, 255), -1)
    cv2.rectangle(vis, (12, 12), (12 + box_w, 12 + pad * 2 + lh * len(lines)), color, 2)
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (28, 12 + pad + 24 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    return vis


def save(path, img):
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return path
