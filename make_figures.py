"""生成报告用的三张图：预测散点、风险—覆盖曲线、方法对比。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "figures"


def read_results():
    with open(ROOT / "out" / "results.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_summary():
    with open(ROOT / "out" / "summary.csv", encoding="utf-8-sig") as f:
        return {r["metric"]: r["value"] for r in csv.DictReader(f)}


def fig_scatter(rows):
    pred = np.array([float(r["pred"]) for r in rows])
    cer = np.array([float(r["cer"]) for r in rows])
    thr = float(read_summary().get("threshold", 0.19))

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    ax.scatter(pred, cer, s=26, alpha=0.75, edgecolor="white", linewidth=0.4)
    lim = max(0.05, float(max(pred.max(), cer.max())) * 1.05)
    ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=1, label="perfect")
    ax.axvline(thr, color="#C0504D", linewidth=1.2, label=f"threshold={thr:.2f}")
    ax.axhline(0.10, color="#1F4E79", linewidth=1.0, linestyle=":", label="limit 10%")
    ax.set_xlabel("predicted CER")
    ax.set_ylabel("true CER")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, min(1.0, lim))
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Prediction vs ground truth", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_scatter.png")
    plt.close(fig)


def fig_risk_coverage(rows):
    pred = np.array([float(r["pred"]) for r in rows])
    cer = np.array([float(r["cer"]) for r in rows])
    order = np.argsort(pred)
    cov, mean, bad = [], [], []
    for i in range(1, len(rows) + 1):
        sel = order[:i]
        cov.append(i / len(rows))
        mean.append(cer[sel].mean())
        bad.append((cer[sel] > 0.10).mean())

    fig, ax1 = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    ax1.plot(cov, mean, color="#1F4E79", label="accepted mean CER")
    ax1.set_xlabel("coverage (fraction accepted)")
    ax1.set_ylabel("mean CER", color="#1F4E79")
    ax1.tick_params(axis="y", labelcolor="#1F4E79")
    ax2 = ax1.twinx()
    ax2.plot(cov, bad, color="#C0504D", linestyle="--", label="bad-accept rate")
    ax2.set_ylabel("fraction with CER > 10%", color="#C0504D")
    ax2.tick_params(axis="y", labelcolor="#C0504D")
    ax1.set_title("Risk–coverage trade-off", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_risk_coverage.png")
    plt.close(fig)


def fig_methods():
    s = read_summary()
    methods = ["b0", "b1", "b2", "ours", "a3"]
    labels = ["B0 sharpness", "B1 OCR conf", "B2 raw pred", "Ours", "A3 conf-only"]
    acc = [float(s.get(f"{m}_accept_rate", 0) or 0) for m in methods]
    cer = [float(s.get(f"{m}_accepted_cer", 0) or 0) for m in methods]
    bad = [float(s.get(f"{m}_bad_accept_rate", 0) or 0) for m in methods]

    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    ax1.bar(x - 0.22, acc, width=0.42, color="#9DB9D4", label="accept rate")
    ax1.set_ylabel("accept rate")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(x, cer, "o-", color="#1F4E79", label="accepted CER")
    ax2.plot(x, bad, "s--", color="#C0504D", label="bad-accept rate")
    ax2.set_ylabel("error rate")
    ax2.set_ylim(0, max(0.35, max(cer + bad) * 1.2))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=12)
    ax1.set_title("Methods: coverage vs risk", fontsize=10)
    fig.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.88))
    fig.tight_layout()
    fig.savefig(OUT / "fig3_methods.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_results()
    if not rows:
        raise SystemExit("out/results.csv 为空，先运行 python run_experiments.py")
    fig_scatter(rows)
    fig_risk_coverage(rows)
    fig_methods()
    print("图已生成：", OUT)


if __name__ == "__main__":
    main()
