# Facial Keypoints Detection Mini Project

本仓库按 `Mini Project Instruction.pdf` 的要求，围绕 Kaggle 赛题 **Facial Keypoints Detection** 构建了一个可复现的端到端项目：

- 数据读取与预处理（含缺失关键点标签掩码）
- Baseline / Improved 两种 CNN 模型（Improved 默认使用更深网络）
- K-Fold 训练与验证（默认 5-fold）
- 训练时随机水平翻转增强 + 推理时水平翻转 TTA
- 基于 `IdLookupTable.csv` 的提交文件生成

## 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 数据准备

从 Kaggle 下载以下文件并放到 `data/` 目录：

- `training.csv`
- `test.csv`
- `IdLookupTable.csv`

目录结构示例：

```text
facial-keypoints-detection/
├── data/
│   ├── training.csv
│   ├── test.csv
│   └── IdLookupTable.csv
├── artifacts/
├── train.py
└── predict.py
```

## 3. 训练（含交叉验证）

默认使用 improved 模型 + 5-fold：

```bash
PYTHONPATH=src python train.py --data-dir data --output-dir artifacts
```

可选 baseline 模型：

```bash
PYTHONPATH=src python train.py --data-dir data --output-dir artifacts --model baseline
```

训练后会得到：

- `artifacts/fold_1.pt` ~ `fold_5.pt`
- `artifacts/cv_summary.json`

## 4. 生成 Kaggle 提交文件

```bash
PYTHONPATH=src python predict.py --data-dir data --artifacts-dir artifacts --output artifacts/submission.csv
```

输出文件：

- `artifacts/submission.csv`

## 5. 与课程 Mini Project 要求对齐

已覆盖核心技术流程：

- Task & Metric：多输出图像回归（官方指标 RMSE）
- Data Analysis & Challenges：缺失标签处理、图像归一化
- Validation Strategy：统一 5-fold CV
- Baseline + Improvements：提供基础与改进模型
- Reproducibility：固定随机种子、明确依赖与运行命令

> 注：最终课程提交仍需补充实验对比、可视化分析、失败尝试与反思，并按课程模板完成 Final Report。
