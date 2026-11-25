"""生成文档页面（中英混排），并给出逐字真值。

真值就是“画上去的文字”，所以 OCR 错误率可以精确计算。
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAGE_W, PAGE_H = 1240, 1754          # A4 @ 150 dpi
MARGIN = 96
FONT_CN = r"C:\Windows\Fonts\msyh.ttc"
FONT_CN_B = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_EN = r"C:\Windows\Fonts\arial.ttf"

CN = [
    "本报告汇总了二〇二六年第三季度的设备巡检结果，共检查点位一百二十处。",
    "现场发现的主要问题包括墙面渗水、管线松动以及标识牌脱落，均已登记在册。",
    "为保证数据可追溯，每一条记录都保留了拍摄时间、设备编号与操作人员。",
    "检查组建议在下月十五日前完成整改，并将结果上传至统一管理平台。",
    "本季度平均响应时间为四小时二十分钟，较上季度缩短了约百分之十八。",
    "对于无法现场处理的问题，应当填写延期说明，并注明预计完成时间。",
    "系统会自动比对历史记录，若同一位置连续三次出现同类问题，将触发预警。",
    "所有原始照片需保留十二个月，逾期由管理员统一归档，不得私自删除。",
    "培训计划安排在每周三下午，内容包括设备操作、数据录入与安全规范。",
    "本次抽检覆盖了三个片区，样本量为四百二十份，有效回收率为百分之九十六。",
    "复核发现两处金额录入错误，已按流程作废并重新提交，未影响最终汇总。",
    "请各部门在提交前自行核对单位、数量与金额，避免因口径不一致导致返工。",
    "巡检记录应当字迹清楚、内容完整，不得使用铅笔填写，也不得随意涂改。",
    "如遇设备故障，请先切断电源，再联系值班人员，切勿自行拆卸。",
    "本表所列数据仅用于内部质量分析，不作为对外发布的依据。",
    "为进一步提升效率，下阶段将试点移动端录入，减少纸质单据的流转环节。",
]

EN = [
    "The inspection covered 120 sites across three districts in the third quarter.",
    "Each record keeps the capture time, device identifier and the operator name.",
    "Photos must be retained for twelve months and archived by the administrator.",
    "Any unresolved issue requires a written extension note with a target date.",
    "The average response time decreased by about 18 percent compared with last quarter.",
    "All figures are rounded to two decimal places unless otherwise stated.",
]

HEADERS = ["项目", "数量", "单价", "金额"]
ROWS = [
    ["墙面修补", "12", "180.00", "2160.00"],
    ["管线更换", "8", "320.50", "2564.00"],
    ["标识牌", "24", "45.00", "1080.00"],
    ["清洁服务", "6", "150.00", "900.00"],
]


def _font(path, size):
    return ImageFont.truetype(path, size)


def _wrap(draw, text, font, max_w):
    """按字符/单词折行，返回行列表。"""
    lines, cur = [], ""
    for ch in text:
        test = cur + ch
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def page_paragraph(rng, page_no):
    img = Image.new("RGB", (PAGE_W, PAGE_H), (246, 246, 246))
    d = ImageDraw.Draw(img)
    f_title = _font(FONT_CN_B, 46)
    f_body = _font(FONT_CN, 34)
    f_small = _font(FONT_EN, 26)
    f_footer = _font(FONT_CN, 24)

    truth = []
    title = f"设备巡检季度报告（第 {page_no} 页）"
    d.text((MARGIN, MARGIN), title, font=f_title, fill="black")
    truth.append(title)

    y = MARGIN + 90
    for i in range(rng.randint(7, 9)):
        if rng.random() < 0.2:
            text = rng.choice(EN)
            font = f_small
        else:
            text = rng.choice(CN)
            font = f_body
        for line in _wrap(d, text, font, PAGE_W - 2 * MARGIN):
            if y > PAGE_H - 190:
                break
            d.text((MARGIN, y), line, font=font, fill="black")
            truth.append(line)
            y += 54 if font is f_body else 40
        y += 14

    footer = f"编号 QC-2026-{page_no:03d}    第 {page_no} 页"
    d.text((MARGIN, PAGE_H - 120), footer, font=f_footer, fill="black")
    truth.append(footer)
    return img, "\n".join(truth)


def page_table(rng, page_no):
    img = Image.new("RGB", (PAGE_W, PAGE_H), (246, 246, 246))
    d = ImageDraw.Draw(img)
    f_title = _font(FONT_CN_B, 44)
    f_cell = _font(FONT_CN, 32)
    f_small = _font(FONT_EN, 26)

    truth = []
    title = f"维修费用明细表（第 {page_no} 页）"
    d.text((MARGIN, MARGIN), title, font=f_title, fill="black")
    truth.append(title)
    truth.append("项目 数量 单价 金额")

    x0, y0 = MARGIN, MARGIN + 130
    col_w = [320, 180, 240, 300]
    row_h = 78
    for r, row in enumerate([HEADERS] + ROWS):
        y = y0 + r * row_h
        d.line([(x0, y), (x0 + sum(col_w), y)], fill="black", width=3)
        x = x0
        for c, cell in enumerate(row):
            d.text((x + 14, y + 16), cell, font=f_cell, fill="black")
            truth.append(cell)
            x += col_w[c]
    d.line([(x0, y0 + 5 * row_h), (x0 + sum(col_w), y0 + 5 * row_h)], fill="black", width=3)
    for c in range(5):
        x = x0 + sum(col_w[:c])
        d.line([(x, y0), (x, y0 + 5 * row_h)], fill="black", width=2)

    note = "注：以上金额单位为人民币元，合计 6704.00 元。"
    d.text((MARGIN, y0 + 5 * row_h + 40), note, font=f_cell, fill="black")
    truth.append(note)
    footer = f"编号 TB-2026-{page_no:03d}"
    d.text((MARGIN, PAGE_H - 200), footer, font=f_small, fill="black")
    truth.append(footer)
    return img, "\n".join(truth)


def page_form(rng, page_no):
    img = Image.new("RGB", (PAGE_W, PAGE_H), (246, 246, 246))
    d = ImageDraw.Draw(img)
    f_title = _font(FONT_CN_B, 44)
    f_body = _font(FONT_CN, 32)
    f_small = _font(FONT_EN, 26)

    truth = []
    title = "现场检查记录表"
    d.text((MARGIN, MARGIN), title, font=f_title, fill="black")
    truth.append(title)

    fields = [
        ("检查日期", "2026 年 11 月 22 日"),
        ("检查地点", "教学楼 B 座三层"),
        ("设备编号", f"EQ-{rng.randint(1000, 9999)}"),
        ("检查人员", "张工、李工"),
        ("问题描述", "墙面出现细微裂缝，长度约 30 厘米。"),
        ("处理意见", "建议一周内安排修补并复查。"),
    ]
    y = MARGIN + 120
    for k, v in fields:
        d.text((MARGIN, y), f"{k}：", font=f_body, fill="black")
        d.text((MARGIN + 240, y), v, font=f_body, fill="black")
        truth.append(f"{k}：{v}")
        y += 90

    footer = f"表单号 FM-{page_no:03d}    签名：__________"
    d.text((MARGIN, PAGE_H - 200), footer, font=f_small, fill="black")
    truth.append(footer)
    return img, "\n".join(truth)


LAYOUTS = {"paragraph": page_paragraph}


def make_pages(n_pages=40, seed=7, out_dir="data/clean"):
    """生成 n 页干净文档，返回 [(image_path, truth_path, page_id, layout)]。"""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n_pages + 1):
        layout = rng.choice(list(LAYOUTS))
        img, truth = LAYOUTS[layout](rng, i)
        pid = f"p{i:03d}"
        ipath = out / f"{pid}.png"
        tpath = out / f"{pid}.txt"
        img.save(ipath)
        tpath.write_text(truth, encoding="utf-8")
        rows.append((str(ipath), str(tpath), pid, layout))
    return rows
