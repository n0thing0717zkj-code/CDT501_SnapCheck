"""单张图演示：预测错字率 -> 能不能用 -> 主因 -> 建议 -> 修正后预测。

用法：
  python demo.py data/degraded/p003_glare65_2.jpg
  python demo.py data/phone/p01.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import load
from src.phone_prep import resize_for_model
from src.pipeline import analyze, annotate, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="out/demo")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--phone", action="store_true", help="手机照片：缩放 + 校准")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    model, thr, meta, attributor = load(root / "out" / "models" / "model.joblib")

    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不到图片：{args.image}")

    is_phone = args.phone or "phone" in str(args.image).lower()
    if is_phone:
        img = resize_for_model(img)

    res = analyze(img, model, thr, attributor=attributor)

    if is_phone:
        calib = root / "out" / "models" / "phone_calibration.joblib"
        if calib.exists():
            iso = joblib.load(calib)["iso"]
            cal = float(np.clip(iso.predict([res["pred_cer"]])[0], 0, 1))
            res["pred_cer_raw"] = res["pred_cer"]
            res["pred_cer"] = round(cal, 4)
            res["accept"] = bool(cal <= 0.10)
            res["status"] = "accept" if res["accept"] else "reject"
            res["top_factor"] = None
            res["advice"] = "可以用" if res["accept"] else "拍得不够清楚，建议重拍"
            res["pred_after_fix"] = round(cal, 4)
            res["threshold"] = 0.10
            thr = 0.10

    vis = annotate(img, res)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    png = out_dir / f"{stem}_result.png"
    save(str(png), vis)

    if args.json:
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        print(json.dumps(clean, ensure_ascii=False, indent=2))
    else:
        print("=" * 52)
        print(f"预测错字率 : {res['pred_cer'] * 100:.1f}%   （门槛 {res['threshold'] * 100:.1f}%）")
        print(f"结论       : {'可以使用' if res['accept'] else '不建议使用，请重拍'}")
        if res.get("top_factor"):
            print(f"主要问题   : {res['top_factor']}（置信度 {res['factor_confidence']:.2f}）")
        print(f"建议       : {res['advice']}")
        print(f"按建议修正 : {res['pred_after_fix'] * 100:.1f}%")
        print(f"结果图     : {png}")
        print("=" * 52)


if __name__ == "__main__":
    main()
