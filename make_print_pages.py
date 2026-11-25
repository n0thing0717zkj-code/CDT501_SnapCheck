# -*- coding: utf-8 -*-
"""生成 5 页可打印文档 + 对应的 20 个 txt（每页 4 份）。"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.render import make_pages

PRINT_DIR = ROOT / "data" / "print"
PHONE = ROOT / "data" / "phone"
PDF_ONE = ROOT / "docs" / "print_pages_5.pdf"
PDF_ALL = ROOT / "docs" / "print_pages_5_all.pdf"


def main():
    PRINT_DIR.mkdir(parents=True, exist_ok=True)
    rows = make_pages(5, seed=2026, out_dir=str(PRINT_DIR))

    # 单页 PDF（方便一页一页打印）
    for i, (png, txt, pid, layout) in enumerate(rows, 1):
        img = Image.open(png)
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis("off")
        fig.savefig(ROOT / "docs" / f"print_p{i:02d}.pdf", dpi=200)
        plt.close(fig)

    # 合并成一个 5 页 PDF
    with PdfPages(PDF_ALL) as pdf:
        for i, (png, txt, pid, layout) in enumerate(rows, 1):
            img = Image.open(png)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    # 旧照片挪走，避免和新照片混在一起
    old = PHONE / "_old_photos"
    old.mkdir(exist_ok=True)
    for jpg in list(PHONE.glob("*.jpg")):
        shutil.move(str(jpg), str(old / jpg.name))

    # 写 20 个 txt：每页 4 份，内容一致
    for i, (png, txt, pid, layout) in enumerate(rows, 1):
        text = Path(txt).read_text(encoding="utf-8")
        for s in "abcd":
            (PHONE / f"p{i:02d}{s}.txt").write_text(text, encoding="utf-8")

    print("可打印 PDF：", PDF_ALL)
    print("单页 PDF  ：", ROOT / "docs" / "print_p01.pdf", "... print_p05.pdf")
    print("txt 已写入 ：", PHONE, "共 20 个")
    print("旧照片已移到：", old)


if __name__ == "__main__":
    main()
