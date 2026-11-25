"""一键回归检查：30 秒内回答“代码还是对的吗”。

运行：python check.py
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import ocr
from src.features import extract
from src.ocr import cer, normalize
from src.render import make_pages
from src.degrade import compose


def test_cer():
    assert cer("abc", "abc") == 0
    assert abs(cer("abcd", "abc") - 1 / 3) < 1e-6
    assert normalize("A b，C。") == "abc"
    print("  [1/4] CER 计算            OK")


def test_render_and_degrade():
    rows = make_pages(1, seed=3, out_dir=str(ROOT / "data" / "_check"))
    img = cv2.imread(rows[0][0])
    assert img is not None and img.shape[0] > 1000
    clean = extract(img)
    blurred, _ = compose(img, "blur", 0.8, None, 0, __import__("random").Random(0))
    b = extract(blurred)
    assert b["lap_var"] < clean["lap_var"], "模糊后清晰度没有下降"
    print("  [2/4] 渲染与退化          OK")


def test_ocr():
    img = np.full((120, 520, 3), 246, np.uint8)
    cv2.putText(img, "CDT501 OCR 2026", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4, cv2.LINE_AA)
    ocr.configure()
    text, conf, lang = ocr.run(img, psm=7)
    assert "CDT" in text.upper() or "501" in text, f"OCR 结果异常：{text!r}"
    print(f"  [3/4] OCR 可用            OK（{lang}, conf={conf:.0f}）")


def test_model():
    path = ROOT / "out" / "models" / "model.joblib"
    if not path.exists():
        print("  [4/4] 模型                SKIP（还没训练）")
        return
    from src.model import load, predict
    model, thr, meta, attr = load(path)
    rows = make_pages(1, seed=9, out_dir=str(ROOT / "data" / "_check"))
    img = cv2.imread(rows[0][0])
    p = predict(model, extract(img))
    assert 0.0 <= p <= 1.0, p
    print(f"  [4/4] 模型可加载          OK（干净页预测 {p * 100:.1f}%）")


def main():
    t0 = time.time()
    print("check.py — 回归检查")
    test_cer()
    test_render_and_degrade()
    test_ocr()
    test_model()
    print(f"PASS  （{time.time() - t0:.1f} 秒）")


if __name__ == "__main__":
    main()
