"""从 CSV 现场训练模型，避免 joblib/sklearn 版本不匹配导致部署失败。"""
import csv
from pathlib import Path

import numpy as np

from src.attribute import train_attributor
from src.features import FEATURES, to_vector
from src.model import choose_threshold, train_model

TAU, ALPHA = 0.10, 0.10


def _rows(root):
    feat = {}
    with open(Path(root) / "data" / "features.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            feat[r["image"]] = {k: float(v) for k, v in r.items() if k != "image"}
    rows = []
    with open(Path(root) / "data" / "labels.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["image"] in feat:
                rows.append({**r, "cer": float(r["cer"]), "features": feat[r["image"]]})
    return rows


def build_from_csv(root):
    rows = _rows(root)
    train = [r for r in rows if r["split"] == "train"]
    calib = [r for r in rows if r["split"] == "calib"]
    if not train or not calib:
        raise RuntimeError("training data missing")

    model = train_model(train, kind="rf")
    p_cal = [float(np.clip(model.predict(to_vector(r["features"]).reshape(1, -1))[0], 0, 1)) for r in calib]
    thr, cov = choose_threshold(p_cal, [r["cer"] for r in calib], tau=TAU, alpha=ALPHA)
    attributor = train_attributor(train)
    meta = {"source": "csv-retrain", "n_train": len(train), "n_calib": len(calib), "accept": cov}
    return model, float(thr), meta, attributor
