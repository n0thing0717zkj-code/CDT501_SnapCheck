"""照片体检 SnapCheck · 本地网页演示 / Local web demo

启动 / Run：双击 启动网页演示.bat，或
             D:\\python\\python3.14.5\\python.exe -m streamlit run app.py
"""
import csv
import sys
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BOOT_ERROR = None
try:
    import cv2
    import joblib
    import numpy as np
    from src.model import load
    from src.phone_prep import resize_for_model
    from src.pipeline import analyze, annotate
except Exception:
    BOOT_ERROR = traceback.format_exc()

st.set_page_config(page_title="照片体检 SnapCheck", page_icon="📷", layout="wide")

if BOOT_ERROR:
    import platform

    st.error("启动失败 / Startup failed")
    st.code(BOOT_ERROR)
    st.caption(
        f"Python {platform.python_version()} · {platform.platform()} · "
        f"cwd={Path.cwd()}"
    )
    st.stop()

CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
.big-card { border-radius: 16px; padding: 22px 26px; border: 1px solid #D6E2EE; background: #F7FAFD; }
.big-num { font-size: 64px; font-weight: 800; line-height: 1.05; margin: 2px 0 6px 0; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 15px; font-weight: 700; }
.muted { color: #6B7A8A; font-size: 13px; }
.hr { height: 1px; background: #E1E9F1; margin: 14px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_model():
    try:
        return load(ROOT / "out" / "models" / "model.joblib")
    except Exception:
        from src.deploy_model import build_from_csv
        return build_from_csv(ROOT)


@st.cache_resource
def get_calibration():
    p = ROOT / "out" / "models" / "phone_calibration.joblib"
    if not p.exists():
        return None
    try:
        return joblib.load(p)["iso"]
    except Exception:
        return None


def read_summary():
    p = ROOT / "out" / "summary.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8-sig") as f:
        return {r["metric"]: r["value"] for r in csv.DictReader(f)}


def decode_upload(uploaded):
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def run(img_bgr, is_phone):
    model, thr, meta, attributor = get_model()
    iso = get_calibration()

    img = resize_for_model(img_bgr) if is_phone else img_bgr
    res = analyze(img, model, thr, attributor=attributor)

    if is_phone and iso is not None:
        cal = float(np.clip(iso.predict([res["pred_cer"]])[0], 0, 1))
        res["pred_cer_raw"] = res["pred_cer"]
        res["pred_cer"] = cal
        res["threshold"] = 0.10
        res["accept"] = bool(cal <= 0.10)
        res["top_factor"] = None
        res["advice"] = ("可以直接使用 / Usable"
                         if res["accept"] else
                         "拍得不够清楚，建议重拍 / Not clear enough, please retake")
        res["pred_after_fix"] = cal
    return img, res


EXAMPLES = {
    "手机 · 正拍 / phone, upright": ("assets/sample_upright.jpg", True),
    "手机 · 斜拍 30° / phone, tilted": ("assets/sample_tilted.jpg", True),
    "手机 · 反光 / phone, glare": ("assets/sample_glare.jpg", True),
    "手机 · 暗光 / phone, dim light": ("assets/sample_dim.jpg", True),
    "合成 · 倾斜 65% / synthetic, tilt": ("assets/synthetic_tilt.jpg", False),
    "合成 · 反光 85% / synthetic, glare": ("assets/synthetic_glare.jpg", False),
}


def main():
    st.markdown("## 照片体检 · SnapCheck")
    st.markdown(
        '<div class="muted">拍完先问一句：这张照片值得拿去识别吗？ / '
        "After the shot, ask: is this photo worth OCR?</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.sidebar:
        st.markdown("### 选择图片 / Choose an image")
        mode = st.radio("来源 / Source", ["快速示例 / Examples", "上传图片 / Upload"],
                        horizontal=True)
        example = None
        uploaded = None
        if mode == "快速示例 / Examples":
            example = st.selectbox("示例 / Example", list(EXAMPLES.keys()))
        else:
            uploaded = st.file_uploader("拖一张图片进来 / Drag an image here",
                                        type=["jpg", "jpeg", "png"])

        st.markdown("---")
        phone_mode = st.radio(
            "图片类型 / Image type",
            ["手机实拍 / Phone photo", "合成图 / Synthetic"],
            index=0,
            help="手机实拍会缩放并做校准。 / Phone photos are resized and calibrated.",
        )
        st.markdown("---")
        st.markdown(
            '<div class="muted">合成数据 300 张 · 手机实拍 20 张 / 300 synthetic · 20 phone photos<br>'
            "归因只在合成数据上验证；手机照片只给“能不能用”。 / "
            "Attribution is validated on synthetic data only; phone photos get accept/reject.</div>",
            unsafe_allow_html=True,
        )

    if uploaded is not None:
        img_bgr = decode_upload(uploaded)
        is_phone = phone_mode == "手机实拍 / Phone photo"
        name = uploaded.name
    elif example is not None:
        rel, is_phone = EXAMPLES[example]
        img_bgr = read_image(ROOT / rel)
        name = Path(rel).name
    else:
        st.info("左侧选一个示例，或上传一张照片。 / Pick an example or upload a photo.")
        return

    if img_bgr is None:
        st.error("这张图读不出来，换一张试试。 / Could not read this image.")
        return

    try:
        img_model, res = run(img_bgr, is_phone)
    except Exception as exc:
        st.error(f"分析失败 / Analysis failed: {type(exc).__name__}: {exc}")
        st.exception(exc)
        return
    accept = res["accept"]
    color = "#2E7D32" if accept else "#C0504D"
    badge_text = "可以使用 / Accept" if accept else "建议重拍 / Retake"
    factor = res.get("top_factor")
    advice = res.get("advice", "")

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.image(cv2.cvtColor(img_model, cv2.COLOR_BGR2RGB), caption=name, width="stretch")

    with right:
        st.markdown(
            f"""
            <div class="big-card">
              <div class="muted">预测错字率 / Predicted CER (wrong characters per 100)</div>
              <div class="big-num" style="color:{color}">{res['pred_cer'] * 100:.1f}%</div>
              <span class="badge" style="background:{color}1A;color:{color}">{badge_text}</span>
              <div class="hr"></div>
              <div class="muted">门槛 / threshold {res['threshold'] * 100:.0f}% ｜ 主要问题 / main cause：{factor or '无 / none'}</div>
              <div style="margin-top:6px;font-size:15px">{advice}</div>
              <div class="muted" style="margin-top:8px">修正后预测 / after-fix estimate：{res.get('pred_after_fix', res['pred_cer']) * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if is_phone and "pred_cer_raw" in res:
            st.caption(
                f"未校准的原始预测 / raw prediction before calibration：{res['pred_cer_raw'] * 100:.1f}%"
                "（合成训练 → 实拍存在域偏移，已用 20 张实拍做留一页校准 / "
                "synthetic-to-camera gap, calibrated leave-one-page-out）"
            )

    tab1, tab2, tab3 = st.tabs(["标注图 / Overlay", "实验结果 / Results", "风险—覆盖 / Risk–coverage"])

    with tab1:
        vis = annotate(img_model, res)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="stretch")

    with tab2:
        summary = read_summary()
        if summary:
            rows = []
            for m, label in [("b0", "B0 清晰度阈值 / sharpness"),
                             ("b1", "B1 OCR 置信度 / confidence"),
                             ("b2", "B2 未校准 / raw predicted"),
                             ("ours", "本方法 / ours"),
                             ("a3", "A3 仅置信度 / confidence-only")]:
                rows.append({
                    "方法 / Method": label,
                    "放行比例 / Accept": summary.get(f"{m}_accept_rate", ""),
                    "放行图错字率 / Accepted CER": summary.get(f"{m}_accepted_cer", ""),
                    "坏图误收 / Bad accepts": summary.get(f"{m}_bad_accept_rate", ""),
                })
            st.markdown("**合成数据（300 张，测试集 72 张） / 300 synthetic, 72 held out**")
            st.dataframe(rows, width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            for col, fn, cap in [
                (c1, "fig1_scatter.png", "预测 vs 真实 / predicted vs true"),
                (c2, "fig3_methods.png", "五种方法对比 / five methods"),
            ]:
                p = ROOT / "out" / "figures" / fn
                if p.exists():
                    col.image(str(p), caption=cap, width="stretch")
        else:
            st.info("还没有实验结果 / No results yet. 先运行 一键跑全量.bat")

    with tab3:
        p = ROOT / "out" / "figures" / "fig2_risk_coverage.png"
        if p.exists():
            st.image(str(p), caption="少收一部分照片，误差能降到多少 / how far the error falls as coverage drops",
                     width="stretch")
        else:
            st.info("还没有风险—覆盖曲线。 / No risk–coverage curve yet.")


if __name__ == "__main__":
    main()
