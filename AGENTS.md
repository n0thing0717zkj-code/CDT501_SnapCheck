# AGENTS.md — 给编码 agent 的规则

## 这个项目是什么

文档照片体检：预测 OCR 错字率、校准接受门槛、归因主因、给重拍建议，
并用“修正后预测”验证建议。报告见 `docs/`。

## 硬规则

1. **不许修改** `check.py`、`tests/`、`data/labels.csv`、`data/phone/*.txt`。
2. 每次改完代码，必须运行 `python check.py`，全绿才算完成。
3. 不许为了让检查通过而修改检查本身。测试错了要说明理由并记录。
4. 不许把测试集（`split == "test"`）或手机实拍数据用于训练/调参/定门槛。
5. 所有结果必须能由 `make_dataset.py → train_model.py → run_experiments.py` 复现。

## 工作方式

- 改代码前先 `git status`，改完看 diff；
- 一次只改一个模块，改完立刻跑 check；
- 新增依赖要写进 `requirements.txt`；
- 中文注释，英文函数名；
- 任何“提升指标”的改动，必须同时报告接受率的变化（不能只报误差）。

## 不许做的事

- 不要引入深度学习 OCR 或云端 API（本项目刻意只用 Tesseract）；
- 不要偷偷调整阈值来美化结果；阈值必须由校准集决定；
- 不要在报告里写没跑过的数字；
- 不要把“合成退化”的结果说成“真实拍摄”的结果。

## 常用命令

```powershell
python check.py
python make_dataset.py --pages 24 --per-page 4
python train_model.py
python run_experiments.py
python make_figures.py
python demo.py data/degraded/xxx.jpg
```
