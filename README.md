# CDT501 · 照片体检 SnapCheck（V4）

**一句话**：一张文档照片，先预测它会被 OCR 认错多少，再告诉你为什么、该怎么重拍，并用“修正后预测”证明建议有效。

不是“检测模糊/反光”的打分器，而是：

1. **预测后果**：回归预测字符错误率（CER），不需要知道正确文本；
2. **校准决策**：用校准集定门槛，说“接受”，就给一个错字率保证；
3. **归因 + 建议**：用质量特征训练归因器，指出主因，给出可执行的一句话；
4. **闭环验证**：把建议对应的修正真的做一遍，看预测错字率是否下降。

---

## 目录

```
src/
  render.py      生成中英混排文档页（真值已知）
  degrade.py     五种退化：模糊/反光/倾斜/阴影/噪声
  ocr.py         Tesseract 封装 + CER
  features.py    14 个无参考质量特征
  model.py       回归 + 门槛校准 + 风险-覆盖
  attribute.py   归因器 + 五种修正 + 建议
  pipeline.py    端到端
make_dataset.py  生成数据 + 跑 OCR 得真值
train_model.py   训练 + 校准 + 保存
run_experiments.py  B0/B1/B2/Ours/A1/A3 + 归因 + 建议有效性
make_figures.py  三张图
demo.py          单张图演示
check.py         30 秒回归检查
tessdata/        eng + chi_sim（Tesseract 语言包）
data/            clean / degraded / phone
out/             models / figures / demo / results.csv
docs/            执行细则 + 组员说明
```

## 环境

```powershell
pip install opencv-python numpy scipy scikit-learn pandas pillow pytesseract matplotlib joblib
# Tesseract 5.x（Windows 安装包）
# https://github.com/UB-Mannheim/tesseract
# 中文语言包放在 tessdata/chi_sim.traineddata（本仓库已带）
```

如果 Tesseract 不在默认路径，设置：

```powershell
$env:TESSERACT_EXE="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 复现

```powershell
python check.py                          # 30 秒自检
python make_dataset.py --pages 24 --per-page 4   # 生成 120 张 + OCR
python train_model.py                    # 训练 + 定门槛
python run_experiments.py                # 对比与消融
python make_figures.py                   # 三张图
python demo.py data/degraded/xxx.jpg     # 单张演示
```

全量建议：`--pages 60 --per-page 5`（约 300 张，OCR 约 6 分钟）。

## 指标

| 指标 | 定义 | 目标 |
|---|---|---|
| MAE | 预测 CER 与真实 CER 的平均绝对差 | ≤ 0.08 |
| 接受图 CER | 通过门槛的图片的真实平均 CER | ≤ 0.10 |
| 坏图误收率 | CER > 10% 却被接受的比例 | ≤ 0.10 |
| 归因准确率 | 主因判断正确比例 | ≥ 0.75 |
| 建议有效率 | 按建议修正后 CER 达标比例 | ≥ 0.60 |

## 手机实拍

把照片放进 `data/phone/`，每张图配一个**同名 .txt**（原文）：

```
data/phone/p01.jpg
data/phone/p01.txt
```

然后：

```powershell
python make_dataset.py --phone
```

`p01.txt` 的内容就是照片里的文字（可直接从原始 Word/PDF 复制）。这份数据只用于**外部验证**，不参与训练。

## 边界

- 单栏文档、中文+英文+数字；不做表格、手写、公式；
- 一种 OCR 引擎（Tesseract 5），不比较引擎优劣；
- 退化以合成为主，实拍 20 张作为外部检验；
- 不用于安全鉴定、票据合规等正式场景。

## 网页演示

```powershell
.\启动网页演示.bat
```

浏览器打开 http://localhost:8501 ：左侧选示例或拖一张图，右侧出结果卡，
下面三个标签页分别是标注图、实验结果、风险—覆盖曲线。

命令行兜底：`演示_三步.bat`（三个固定案例）或把图片拖到 `演示_拖图.bat`。

手机照片会自动缩放到训练尺寸并做留一页校准；合成图用原模型。

## 部署到公网（Streamlit Community Cloud）

1. 打开 https://share.streamlit.io ，用 GitHub 登录并授权访问 `CDT501_SnapCheck` 仓库
2. New app → 选择 `n0thing0717zkj-code/CDT501_SnapCheck`，branch `main`，main file `app.py`
3. Advanced settings → Python 3.12
4. Deploy（首次约 3–5 分钟安装依赖和 Tesseract）
5. 得到公开链接，任何人可打开；免费实例闲置会休眠，首次访问需等 20–30 秒唤醒

仓库里的 `packages.txt` 会在服务器上安装 `tesseract-ocr`，`tessdata/` 提供中英文语言包。


