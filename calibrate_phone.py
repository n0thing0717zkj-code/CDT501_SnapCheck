"""用手机实拍图校准预测值（isotonic），并报告留一页（LOO）结果。

为什么需要：模型只用合成数据训练，预测值在真实照片上被压缩在 0.4 附近，
绝对数值偏高。用 20 张实拍做一个单调校准，把数值拉回真实量级。

用法：python calibrate_phone.py
输出：out/models/phone_calibration.joblib + 控制台 LOO 指标
"""
import csv
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features import extract
from src.model import load, predict
from src.phone_prep import resize_for_model

TAU = 0.10


def load_pairs():
    labels = {r["page_id"].replace("phone_", ""): float(r["cer"])
              for r in csv.DictReader(open(ROOT / "data" / "phone_labels.csv", encoding="utf-8-sig"))}
    model, thr, meta, attr = load(ROOT / "out" / "models" / "model.joblib")
    pairs = []
    for p in sorted((ROOT / "data" / "phone").glob("*.jpg")):
        if p.stem not in labels:
            continue
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        pairs.append((p.stem, predict(model, extract(resize_for_model(img))), labels[p.stem]))
    return pairs


def loo(pairs):
    pages = sorted({n[:3] for n, _, _ in pairs})
    out = {}
    for pg in pages:
        cal = [(p, y) for n, p, y in pairs if n[:3] != pg]
        xs = np.array([p for p, _ in cal]); ys = np.array([y for _, y in cal])
        iso = IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
        for n, p, _ in pairs:
            if n[:3] == pg:
                out[n] = float(np.clip(iso.predict([p])[0], 0, 1))
    return out


def main():
    pairs = load_pairs()
    if len(pairs) < 10:
        raise SystemExit("实拍样本太少，先跑 make_dataset.py --phone")

    pred = loo(pairs)
    truth = {n: y for n, _, y in pairs}
    errs = [abs(pred[n] - truth[n]) for n in pred]
    acc = np.array([pred[n] <= TAU for n in pred])
    cer = np.array([truth[n] for n in pred])
    print(f"LOO: n={len(pred)}  MAE={np.mean(errs):.3f}  accept={acc.mean():.2f}  "
          f"accepted_cer={cer[acc].mean() if acc.any() else float('nan'):.3f}  "
          f"bad={(cer[acc] > TAU).mean() if acc.any() else float('nan'):.2f}")

    xs = np.array([p for _, p, _ in pairs]); ys = np.array([y for _, _, y in pairs])
    iso = IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
    out = ROOT / "out" / "models" / "phone_calibration.joblib"
    joblib.dump({"iso": iso, "tau": TAU, "n": len(pairs)}, out)
    print("校准已保存：", out)


if __name__ == "__main__":
    main()
