"""特征消融：逐个去掉一个特征 / 一整组，看测试误差怎么变。

用法：python feature_ablation.py
输出：out/feature_ablation.csv、out/figures/fig5_feature_ablation.png、out/feature_ablation.md
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features import FEATURES
from src.model import choose_threshold, evaluate

TAU, ALPHA = 0.10, 0.10
SEED = 0

GROUPS = {
    "sharpness": ["lap_var", "tenengrad", "hf_ratio", "text_height"],
    "illumination": ["bright_frac", "glare_blob_frac", "glare_contrast", "illum_gradient"],
    "geometry": ["tilt_deg"],
    "noise": ["noise_sigma", "blockiness", "contrast", "mean", "std"],
}


def load():
    feat = {}
    with open(ROOT / "data" / "features.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            feat[r["image"]] = {k: float(v) for k, v in r.items() if k != "image"}
    rows = []
    with open(ROOT / "data" / "labels.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["image"] in feat:
                rows.append({**r, "cer": float(r["cer"]), "features": feat[r["image"]]})
    return rows


def train(rows, cols):
    X = np.array([[r["features"][c] for c in cols] for r in rows])
    y = np.array([r["cer"] for r in rows])
    m = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                              max_features=0.7, random_state=SEED, n_jobs=-1)
    m.fit(X, y)
    return m, cols


def pred(m, cols, rows):
    X = np.array([[r["features"][c] for c in cols] for r in rows])
    return np.clip(m.predict(X), 0, 1)


def run(rows, drop, label, kind):
    cols = [c for c in FEATURES if c not in drop]
    train_rows = [r for r in rows if r["split"] == "train"]
    calib = [r for r in rows if r["split"] == "calib"]
    test = [r for r in rows if r["split"] == "test"]

    m, cols = train(train_rows, cols)
    p_cal = pred(m, cols, calib)
    thr, _ = choose_threshold(p_cal, [r["cer"] for r in calib], tau=TAU, alpha=ALPHA)
    p_test = pred(m, cols, test)
    met = evaluate(p_test, [r["cer"] for r in test], thr, tau=TAU)
    return {
        "removed": label,
        "kind": kind,
        "n_features": len(cols),
        "mae": round(met["mae"], 4),
        "accept": round(met["accept_rate"], 3),
        "acc_cer": round(met["accepted_cer"], 4) if met["accepted_cer"] == met["accepted_cer"] else "",
        "bad": round(met["bad_accept_rate"], 3) if met["bad_accept_rate"] == met["bad_accept_rate"] else "",
        "threshold": round(thr, 4),
    }


def main():
    rows = load()
    if not rows:
        raise SystemExit("没有数据，先跑 make_dataset.py")

    base = run(rows, [], "none (all 14)", "baseline")
    out = [base]
    for f in FEATURES:
        out.append(run(rows, [f], f, "single"))
    for g, cols in GROUPS.items():
        out.append(run(rows, cols, f"[group] {g}", "group"))

    b = base["mae"]
    for r in out:
        r["d_mae"] = round(r["mae"] - b, 4)

    outdir = ROOT / "out"
    outdir.mkdir(exist_ok=True)
    cols = ["removed", "kind", "n_features", "mae", "d_mae", "accept", "acc_cer", "bad", "threshold"]
    with open(outdir / "feature_ablation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    rows_fig = [r for r in out if r["kind"] != "baseline"]
    rows_fig.sort(key=lambda r: r["d_mae"], reverse=True)
    labels = [r["removed"].replace("[group] ", "") for r in rows_fig]
    vals = [r["d_mae"] for r in rows_fig]
    colors = ["#A6192E" if r["kind"] == "group" else "#9DB9D4" for r in rows_fig]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("increase in test MAE when removed", fontsize=9)
    ax.set_title(f"Feature ablation (baseline MAE = {b:.3f})", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    (ROOT / "out" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "out" / "figures" / "fig5_feature_ablation.png")
    plt.close(fig)

    # markdown summary
    top = rows_fig[:3]
    lines = [
        "# Feature ablation",
        "",
        f"Baseline (all 14 features): MAE = {b:.3f}, accept = {base['accept']:.2f}, "
        f"accepted CER = {base['acc_cer']:.3f}, bad accepts = {base['bad']:.2f}.",
        "",
        "| Removed | Kind | MAE | ΔMAE | Accept | Accepted CER |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows_fig:
        lines.append(f"| {r['removed']} | {r['kind']} | {r['mae']:.3f} | {r['d_mae']:+.3f} | "
                     f"{r['accept']:.2f} | {r['acc_cer']} |")
    lines += ["", "Largest increases:"]
    for r in top:
        lines.append(f"- {r['removed']}: +{r['d_mae']:.3f} MAE")
    (outdir / "feature_ablation.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"baseline MAE {b:.3f}   accept {base['accept']:.2f}   CER {base['acc_cer']}")
    print(f"{'removed':24s} {'kind':8s} {'MAE':>6s} {'dMAE':>7s} {'accept':>7s}")
    for r in rows_fig:
        print(f"{r['removed']:24s} {r['kind']:8s} {r['mae']:6.3f} {r['d_mae']:+7.3f} {r['accept']:7.2f}")
    print("saved: out/feature_ablation.csv, out/feature_ablation.md, out/figures/fig5_feature_ablation.png")


if __name__ == "__main__":
    main()
