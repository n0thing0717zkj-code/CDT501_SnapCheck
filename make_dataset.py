"""生成数据集：干净页 -> 退化图 -> OCR -> 真实错字率。

用法：
  python make_dataset.py --pages 40 --per-page 5
  python make_dataset.py --phone            # 只处理 data/phone 里的实拍图
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import ocr
from src import phone_prep
from src.degrade import FACTORS, capture_effects, compose
from src.render import make_pages

TAU = 0.02


def _save(path, bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    buf.tofile(str(path))


def build_pages(root, n_pages, per_page, seed):
    rng = random.Random(seed)
    rows = make_pages(n_pages=n_pages, seed=seed, out_dir=root / "data" / "clean")
    out_rows = []

    for ipath, tpath, pid, layout in rows:
        bgr = cv2.imread(ipath)
        truth = Path(tpath).read_text(encoding="utf-8")
        frac = int(pid[1:]) / max(1, n_pages)
        split = "train" if frac <= 0.6 else ("calib" if frac <= 0.8 else "test")

        # 干净版本（也加相机真实感，模拟“拍得很好的手机照片”）
        clean_img = capture_effects(bgr, rng) if rng.random() < 0.8 else bgr
        clean_path = root / "data" / "degraded" / f"{pid}_clean.jpg"
        _save(clean_path, clean_img)
        out_rows.append(dict(image=str(clean_path), page_id=pid, layout=layout, split=split,
                             factor="none", severity=0.0, truth_path=tpath))

        # 退化版本
        for k in range(per_page):
            factor = rng.choice(FACTORS)
            sev = rng.choice([0.25, 0.45, 0.65, 0.85])
            secondary = rng.choice([None] + FACTORS)
            sec_sev = 0.25 if secondary else 0.0
            if secondary == factor:
                secondary, sec_sev = None, 0.0
            img, meta = compose(bgr, factor, sev, secondary, sec_sev, rng)
            if rng.random() < 0.85:
                img = capture_effects(img, rng)
            name = f"{pid}_{factor}{int(sev * 100):02d}_{k}.jpg"
            path = root / "data" / "degraded" / name
            _save(path, img)
            out_rows.append(dict(image=str(path), page_id=pid, layout=layout, split=split,
                                 factor=factor, severity=sev, truth_path=tpath))
    return out_rows


def run_ocr(rows, tau=TAU):
    cfg = ocr.configure()
    lang = ocr.pick_lang()
    print(f"OCR: {cfg['exe']}  tessdata={cfg['tessdata']}  lang={lang}")
    for i, r in enumerate(rows, 1):
        if r.get("cer") is not None:
            continue
        img = cv2.imdecode(np.fromfile(r["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if r.get("split") == "phone":
            # 方向已由人工转正；这里只把尺寸缩放到与训练集一致
            img = phone_prep.resize_for_model(img)
        truth = Path(r["truth_path"]).read_text(encoding="utf-8")
        text, conf, _ = ocr.run(img, lang=lang, psm=6)
        r["cer"] = round(ocr.cer(text, truth), 4)
        r["conf"] = round(conf, 2)
        r["lang"] = lang
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  cer={r['cer']:.3f}")
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["image", "page_id", "layout", "split", "factor", "severity",
            "truth_path", "cer", "conf", "lang"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print("写出", path, len(rows), "行")


def scan_phone(root):
    """data/phone 下每张图配一个同名 .txt（原文）。"""
    rows = []
    for img in sorted((root / "data" / "phone").glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        txt = img.with_suffix(".txt")
        if not txt.exists():
            print("跳过（缺 .txt）：", img.name)
            continue
        rows.append(dict(image=str(img), page_id="phone_" + img.stem, layout="phone",
                         split="phone", factor="real", severity="",
                         truth_path=str(txt)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--per-page", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--phone", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    if args.phone:
        rows = scan_phone(root)
        if not rows:
            print("data/phone 里没有可用的图片+txt，请先拍照并按 p01.jpg / p01.txt 命名。")
            return
        rows = [] if args.no_ocr else run_ocr(rows)
        write_csv(rows, root / "data" / "phone_labels.csv")
        return

    rows = build_pages(root, args.pages, args.per_page, args.seed)
    print(f"生成 {len(rows)} 张图")
    if not args.no_ocr:
        rows = run_ocr(rows)
    write_csv(rows, root / "data" / "labels.csv")


if __name__ == "__main__":
    main()
