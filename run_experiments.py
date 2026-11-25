"""跑对比与消融：B0/B1/B2、Ours、A1/A2/A3，以及归因与建议有效性。

输出 out/results.csv（逐图）与 out/summary.csv（汇总）。
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import ocr
from src.attribute import CORRECT, attribute
from src.features import extract, to_vector
from src.model import choose_threshold, evaluate, load, predict, train_model
from src.ocr import cer as cer_fn
from train_model import TAU, load_features, read_rows


def _tune_high(values, truth, tau, alpha, min_accept=8):
    """特征值越大越好的门槛（如清晰度、置信度）。"""
    v = np.asarray(values, float)
    y = np.asarray(truth, float)
    order = np.argsort(-v)
    best = None
    for k in range(min_accept, len(v) + 1):
        idx = order[:k]
        if float((y[idx] > tau).mean()) <= alpha:
            best = float(v[idx].min())
    return best if best is not None else float(v.max() + 1)


def _heuristic_factor(feat):
    if feat["tilt_deg"] > 8:
        return "tilt"
    if feat["glare_blob_frac"] > 0.01:
        return "glare"
    if feat["illum_gradient"] > 0.15:
        return "shadow"
    if feat["lap_var"] < 0.004:
        return "blur"
    if feat["noise_sigma"] > 3:
        return "noise"
    return None


def main():
    root = Path(__file__).resolve().parent
    rows = read_rows(root / "data" / "labels.csv")
    rows = load_features(rows, root / "data" / "features.csv")
    train = [r for r in rows if r["split"] == "train"]
    calib = [r for r in rows if r["split"] == "calib"]
    test = [r for r in rows if r["split"] == "test"]

    model, thr, meta, attributor = load(root / "out" / "models" / "model.joblib")
    p_cal = [predict(model, r["features"]) for r in calib]
    y_cal = [r["cer"] for r in calib]
    p_test = [predict(model, r["features"]) for r in test]
    y_test = [r["cer"] for r in test]

    b0_thr = _tune_high([r["features"]["lap_var"] for r in calib], y_cal, TAU, 0.1)
    b1_thr = _tune_high([r["conf"] for r in calib], y_cal, TAU, 0.1)

    # A3：只用 OCR 置信度
    from sklearn.ensemble import RandomForestRegressor
    Xc = np.array([[float(r["conf"])] for r in train])
    yc = np.array([r["cer"] for r in train])
    m_conf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                   random_state=0).fit(Xc, yc)
    p_conf_cal = [float(np.clip(m_conf.predict([[r["conf"]]])[0], 0, 1)) for r in calib]
    thr_conf, _ = choose_threshold(p_conf_cal, y_cal, tau=TAU, alpha=0.1)

    results = []
    for r, pred in zip(test, p_test):
        feat = r["features"]
        item = {
            "image": Path(r["image"]).name,
            "factor": r["factor"],
            "severity": r["severity"],
            "cer": r["cer"],
            "conf": r["conf"],
            "pred": round(pred, 4),
        }
        item["ours"] = int(pred <= thr)
        item["b0"] = int(feat["lap_var"] >= b0_thr)
        item["b1"] = int(float(r["conf"]) >= b1_thr)
        item["b2"] = int(pred <= TAU)
        p_conf = float(np.clip(m_conf.predict([[float(r["conf"])]])[0], 0, 1))
        item["a3"] = int(p_conf <= thr_conf)

        img = cv2.imdecode(np.fromfile(r["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        attr = attribute(img, model, attributor=attributor, base_feat=feat)
        item["attr_top"] = attr["top_factor"] or "none"
        item["attr_ok"] = int(attr["top_factor"] == r["factor"]) if r["factor"] != "none" else ""
        # A2：只用反事实（逐个修正后看预测降幅），不做分类器
        best_f, best_d = "none", 0.0
        for f in ("blur", "glare", "tilt", "shadow", "noise"):
            p = predict(model, extract(CORRECT[f](img)))
            if predict(model, feat) - p > best_d:
                best_f, best_d = f, predict(model, feat) - p
        item["cf_top"] = best_f if best_d > 0.004 else "none"
        item["cf_ok"] = int(item["cf_top"] == r["factor"]) if r["factor"] != "none" else ""
        results.append(item)

    # 建议有效性：按“真实主因”做修正，再跑 OCR
    ocr.configure()
    lang = ocr.pick_lang()
    fixed_ok, fixed_n, drop = 0, 0, []
    for r in test:
        f = r["factor"]
        if f == "none":
            continue
        img = cv2.imdecode(np.fromfile(r["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        truth = Path(r["truth_path"]).read_text(encoding="utf-8")
        fixed = CORRECT[f](img)
        text, _, _ = ocr.run(fixed, lang=lang, psm=6)
        new_cer = cer_fn(text, truth)
        fixed_n += 1
        if new_cer <= TAU:
            fixed_ok += 1
        drop.append(r["cer"] - new_cer)

    summary = []
    for m in ("b0", "b1", "b2", "ours", "a3"):
        acc = np.array([x[m] for x in results], bool)
        cer_arr = np.array([x["cer"] for x in results])
        summary.append({
            "method": m,
            "accept_rate": round(float(acc.mean()), 3),
            "accepted_cer": round(float(cer_arr[acc].mean()), 3) if acc.any() else "",
            "bad_accept_rate": round(float((cer_arr[acc] > TAU).mean()), 3) if acc.any() else "",
        })

    attr_rows = [x for x in results if x["attr_ok"] != ""]
    attr_acc = np.mean([x["attr_ok"] for x in attr_rows]) if attr_rows else float("nan")
    cf_acc = np.mean([x["cf_ok"] for x in attr_rows]) if attr_rows else float("nan")

    out = root / "out"
    out.mkdir(exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for s in summary:
            w.writerow([f"{s['method']}_accept_rate", s["accept_rate"]])
            w.writerow([f"{s['method']}_accepted_cer", s["accepted_cer"]])
            w.writerow([f"{s['method']}_bad_accept_rate", s["bad_accept_rate"]])
        w.writerow(["mae_ours", round(float(np.mean(np.abs(np.array(p_test) - np.array(y_test)))), 4)])
        w.writerow(["attribution_accuracy", round(float(attr_acc), 3)])
        w.writerow(["counterfactual_accuracy", round(float(cf_acc), 3)])
        w.writerow(["advice_efficacy", round(fixed_ok / max(fixed_n, 1), 3)])
        w.writerow(["advice_mean_cer_drop", round(float(np.mean(drop)), 3) if drop else ""])
        w.writerow(["n_test", len(test)])
        w.writerow(["threshold", round(thr, 4)])

    print("方法       接受率  接受图CER  坏图率")
    for s in summary:
        print(f"{s['method']:<8} {s['accept_rate']:.2f}    {s['accepted_cer']}      {s['bad_accept_rate']}")
    print(f"MAE={np.mean(np.abs(np.array(p_test)-np.array(y_test))):.3f}  "
          f"归因={attr_acc:.2f}  反事实={cf_acc:.2f}  建议有效={fixed_ok / max(fixed_n,1):.2f}")


if __name__ == "__main__":
    main()
