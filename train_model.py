"""训练质量预测器，并在校准集上定“接受门槛”。

用法：python train_model.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features import extract, to_vector
from src.attribute import train_attributor
from src.model import choose_threshold, evaluate, save, train_model

TAU = 0.10
ALPHA = 0.10


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            r["cer"] = float(r["cer"])
            r["severity"] = float(r["severity"] or 0)
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def load_features(rows, cache_path):
    """特征缓存：同一张图只算一次。"""
    cache = {}
    p = Path(cache_path)
    if p.exists():
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                cache[r["image"]] = {k: float(v) for k, v in r.items() if k != "image"}

    todo = [r for r in rows if r["image"] not in cache]
    for i, r in enumerate(todo, 1):
        img = cv2.imdecode(np.fromfile(r["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        cache[r["image"]] = extract(img)
        if i % 20 == 0:
            print(f"  特征 {i}/{len(todo)}")

    if todo:
        keys = list(cache[rows[0]["image"]].keys())
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["image"] + keys)
            for k, v in cache.items():
                w.writerow([k] + [v[kk] for kk in keys])

    for r in rows:
        r["features"] = cache[r["image"]]
    return rows


def main():
    root = Path(__file__).resolve().parent
    rows = read_rows(root / "data" / "labels.csv")
    if not rows:
        print("labels.csv 为空，先运行 python make_dataset.py")
        return
    rows = load_features(rows, root / "data" / "features.csv")

    train = [r for r in rows if r["split"] == "train"]
    calib = [r for r in rows if r["split"] == "calib"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"train={len(train)}  calib={len(calib)}  test={len(test)}")
    if len(train) < 20 or len(calib) < 8 or len(test) < 8:
        print("样本太少：至少 20/8/8。先把 --pages 调大。")
        return

    model = None
    best_kind, best_mae = None, 9.0
    for kind in ("rf", "hgb"):
        m = train_model(train, kind=kind)
        p = [float(np.clip(m.predict(to_vector(r["features"]).reshape(1, -1))[0], 0, 1)) for r in calib]
        mae = float(np.mean(np.abs(np.array(p) - np.array([r["cer"] for r in calib]))))
        print(f"  {kind}: 校准集 MAE = {mae:.4f}")
        if mae < best_mae:
            model, best_kind, best_mae = m, kind, mae
    print(f"选用 {best_kind}")

    p_cal = [float(np.clip(model.predict(to_vector(r["features"]).reshape(1, -1))[0], 0, 1)) for r in calib]
    y_cal = [r["cer"] for r in calib]
    thr, cov = choose_threshold(p_cal, y_cal, tau=TAU, alpha=ALPHA)
    print(f"门槛 = {thr:.4f}  预计接受比例 = {cov:.2f}")

    p_test = [float(np.clip(model.predict(to_vector(r["features"]).reshape(1, -1))[0], 0, 1)) for r in test]
    y_test = [r["cer"] for r in test]
    m = evaluate(p_test, y_test, thr, tau=TAU)
    print("测试集：", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

    save(model, thr, {"tau": TAU, "alpha": ALPHA, "n_train": len(train),
                      "n_calib": len(calib), "n_test": len(test),
                      "test_metrics": m}, root / "out" / "models" / "model.joblib",
         attributor=train_attributor(train))
    print("模型已保存：out/models/model.joblib")


if __name__ == "__main__":
    main()
