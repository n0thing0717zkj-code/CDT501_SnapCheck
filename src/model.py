"""训练、校准、风险控制。

目标不是“预测得准”，而是“被接受的图，错字率真的有保证”。
"""
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features import FEATURES, to_vector


def train_model(rows, seed=0, kind="rf"):
    X = np.vstack([to_vector(r["features"]) for r in rows])
    y = np.array([r["cer"] for r in rows], dtype=np.float64)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.7,
            random_state=seed, n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=12,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
            random_state=seed,
        )
    model.fit(X, np.clip(y, 0, 1))
    return model


def predict(model, feat):
    return float(np.clip(model.predict(to_vector(feat).reshape(1, -1))[0], 0, 1))


def cp_upper(bad, n, alpha=0.05):
    """Clopper-Pearson 上界：观察到 bad/n，真实比例 95% 上界。"""
    if n <= 0:
        return 1.0
    if bad <= 0:
        return float(1.0 - alpha ** (1.0 / n))
    return float(beta.ppf(1 - alpha, bad + 1, n - bad))


def choose_threshold(pred, truth, tau=0.10, alpha=0.10, min_accept=10):
    """选“接受比例最高且坏图率 ≤ alpha”的门槛。

    做实验时用经验坏图率选阈值；报告里同时给出 Clopper-Pearson 上界，
    说明小样本下这个保证有多大的不确定性。
    """
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    order = np.argsort(pred)
    best_t, best_cov = -np.inf, 0.0
    for k in range(min_accept, len(pred) + 1):
        idx = order[:k]
        bad = float((truth[idx] > tau).mean())
        if bad <= alpha:
            t = float(pred[idx].max())
            if t > best_t:
                best_t, best_cov = t, k / len(pred)
    return best_t, float(best_cov)


def evaluate(pred, truth, threshold, tau=0.02):
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, float)
    accept = pred <= threshold
    out = {
        "n": int(len(pred)),
        "mae": float(np.mean(np.abs(pred - truth))),
        "spearman": float(_spearman(pred, truth)),
        "accept_rate": float(accept.mean()),
        "accepted_cer": float(truth[accept].mean()) if accept.any() else float("nan"),
        "bad_accept_rate": float((truth[accept] > tau).mean()) if accept.any() else float("nan"),
    }
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return 0.0 if np.isnan(r) else float(r)


def risk_coverage(pred, truth, tau=0.02, steps=19):
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, float)
    order = np.argsort(pred)
    out = []
    for i in range(1, steps + 1):
        frac = i / steps
        k = max(1, int(round(len(pred) * frac)))
        sel = order[:k]
        out.append({
            "coverage": frac,
            "mean_cer": float(truth[sel].mean()),
            "bad_rate": float((truth[sel] > tau).mean()),
        })
    return out


def save(model, threshold, meta, model_path="out/models/model.joblib", attributor=None):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "meta": meta,
                 "features": FEATURES, "attributor": attributor}, model_path)
    return model_path


def load(model_path="out/models/model.joblib"):
    d = joblib.load(model_path)
    return d["model"], float(d["threshold"]), d.get("meta", {}), d.get("attributor")
