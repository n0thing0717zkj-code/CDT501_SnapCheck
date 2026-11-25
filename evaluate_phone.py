"""在手机实拍图上评估：预测错字率 vs 真实错字率。

用法：python evaluate_phone.py
输出：out/phone_results.csv + 控制台表格
"""
import csv
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.attribute import attribute
from src.features import extract
from src.model import load, predict
from src.phone_prep import resize_for_model


def main():
    labels_path = ROOT / "data" / "phone_labels.csv"
    if not labels_path.exists():
        raise SystemExit("先运行 python make_dataset.py --phone")

    truth = {r["page_id"].replace("phone_", ""): float(r["cer"])
             for r in csv.DictReader(open(labels_path, encoding="utf-8-sig"))}

    model, thr, meta, attributor = load(ROOT / "out" / "models" / "model.joblib")

    rows = []
    for img_path in sorted((ROOT / "data" / "phone").glob("*.jpg")):
        key = img_path.stem
        if key not in truth:
            continue
        raw = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = resize_for_model(raw)
        feat = extract(img)
        pred = predict(model, feat)
        attr = attribute(img, model, attributor=attributor, base_feat=feat)
        rows.append({
            "image": img_path.name,
            "true_cer": round(truth[key], 4),
            "pred_cer": round(pred, 4),
            "cal_cer": round(pred, 4),
            "abs_err": round(abs(pred - truth[key]), 4),
            "abs_err_cal": round(abs(pred - truth[key]), 4),
            "accept": int(pred <= thr),
            "accept_cal": 0,
            "top_factor": attr["top_factor"] or "none",
            "pred_after_fix": round(attr["pred_after_fix"], 4),
        })

    # 留一页校准：用其他 4 页拟合 isotonic，预测被留出的那一页
    from sklearn.isotonic import IsotonicRegression
    pred_map = {r["image"]: r["pred_cer"] for r in rows}
    for pg in sorted({r["image"][:3] for r in rows}):
        cal = [(pred_map[r["image"]], r["true_cer"]) for r in rows if r["image"][:3] != pg]
        xs = np.array([p for p, _ in cal]); ys = np.array([y for _, y in cal])
        iso = IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
        for r in rows:
            if r["image"][:3] == pg:
                c = float(np.clip(iso.predict([r["pred_cer"]])[0], 0, 1))
                r["cal_cer"] = round(c, 4)
                r["abs_err_cal"] = round(abs(c - r["true_cer"]), 4)
                r["accept_cal"] = int(c <= 0.10)

    if not rows:
        raise SystemExit("没有找到成对的图片/标签")

    out = ROOT / "out" / "phone_results.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'image':12s} {'true':>6s} {'raw':>6s} {'cal':>6s} {'err':>6s} {'acc':>5s} {'factor':>8s}")
    for r in rows:
        print(f"{r['image']:12s} {r['true_cer']:6.3f} {r['pred_cer']:6.3f} {r['cal_cer']:6.3f} "
              f"{r['abs_err_cal']:6.3f} {str(bool(r['accept_cal'])):>5s} {r['top_factor']:>8s}")

    mae = np.mean([r["abs_err"] for r in rows])
    mae_cal = np.mean([r["abs_err_cal"] for r in rows])
    acc = np.array([r["accept_cal"] for r in rows], bool)
    cer = np.array([r["true_cer"] for r in rows])
    print(f"\nn={len(rows)}")
    print(f"未校准：MAE={mae:.3f}  接受率={sum(r['accept'] for r in rows) / len(rows):.2f}")
    print(f"已校准：MAE={mae_cal:.3f}  接受率={acc.mean():.2f}  "
          f"接受图CER={cer[acc].mean() if acc.any() else float('nan'):.3f}  "
          f"坏图误收={(cer[acc] > 0.10).mean() if acc.any() else float('nan'):.2f}")
    print("结果表：", out)


if __name__ == "__main__":
    main()
