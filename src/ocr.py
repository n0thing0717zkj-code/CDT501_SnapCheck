"""Tesseract 封装 + 字符错误率（CER）。

中文和英文混排统一处理：只保留汉字、字母、数字，丢弃空白与标点，
避免“换行位置不同”被误算成识别错误。
"""
import os
import re
from pathlib import Path

import numpy as np
import pytesseract

DEFAULT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESSDATA = _PROJECT_ROOT / "tessdata"

_KEEP = re.compile(r"[^\u4e00-\u9fffa-zA-Z0-9]+")


def configure(tessdata=None, exe=None):
    if exe is None:
        exe = os.environ.get("TESSERACT_EXE", DEFAULT_EXE)
    if os.path.exists(exe):
        pytesseract.pytesseract.tesseract_cmd = exe
    else:
        raise FileNotFoundError(
            "找不到 tesseract.exe。请安装 Tesseract，或设置环境变量 TESSERACT_EXE。"
        )
    data = str(tessdata or DEFAULT_TESSDATA)
    if os.path.isdir(data):
        os.environ["TESSDATA_PREFIX"] = data
    return {"exe": exe, "tessdata": data}


def available_langs():
    return set(pytesseract.get_languages(config=""))


def pick_lang(prefer=("chi_sim", "eng")):
    langs = available_langs()
    use = [l for l in prefer if l in langs]
    return "+".join(use) if use else "eng"


def run(img_bgr, lang=None, psm=6):
    """返回 (文本, 平均置信度, 语言)。"""
    if lang is None:
        lang = pick_lang()
    rgb = img_bgr[:, :, ::-1]
    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(rgb, lang=lang, config=config)
    data = pytesseract.image_to_data(rgb, lang=lang, config=config,
                                     output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(data["conf"], data["text"])
             if t and t.strip() and float(c) >= 0]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return text, mean_conf, lang


def normalize(s):
    return _KEEP.sub("", s).lower()


def edit_distance(a, b):
    """Levenshtein 距离（滚动数组）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, truth):
    p, t = normalize(pred), normalize(truth)
    if not t:
        return float("nan")
    return float(edit_distance(p, t) / len(t))
