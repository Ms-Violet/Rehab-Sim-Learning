# Rehab-Sim-Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-required-red)

本项目用于基于肌肉信号时间序列（多通道 `Top_*`）回归预测物理参数 **K** 与 **C**（标签从文件名解析得到），包含深度学习与传统机器学习两类基线脚本。脚本运行后会输出评估指标并保存模型文件到 `model/`。

## 目录

- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [数据约定（与代码保持一致）](#数据约定与代码保持一致)
- [环境依赖](#环境依赖)
- [运行方式](#运行方式)
- [输出产物](#输出产物)
- [常见问题](#常见问题)
- [贡献](#贡献)
- [许可证](#许可证)

## 快速开始

在项目根目录执行（确保相对路径 `./data1`、`./data2` 可用）：

```bash
pip install numpy pandas matplotlib scikit-learn joblib
pip install torch
pip install xgboost
python ResNet_v2.py
```

## 目录结构

- `data1/`：单文件单样本数据（用于 `plsr.py`、`XGBoost.py`）
- `data2/`：三自由度（F/D/P）多模态数据（用于 `GRU_bidir.py`、`LSTM_bidir.py`、`ResNet.py`、`ResNet_v2.py`、`Inception.py`）
  - `Flexion_p/`：`*_F_p.csv`
  - `Deviation_p/`：`*_D_p.csv`
  - `Pronation_p/`：`*_P_p.csv`
- `model/`：训练得到的权重与归一化器（`.pth`/`.pkl`），以及部分脚本拷贝
- 训练脚本：
  - `GRU_bidir.py`：Bi-GRU（对标签做 `log` 变换）
  - `LSTM_bidir.py`：Bi-LSTM（对标签做标准化/反归一化）
  - `ResNet.py`：1D-ResNet（整体输入归一化 + 目标标准化）
  - `ResNet_v2.py`：Late-Fusion 版 1D-ResNet（将 F/D/P 三路输入分支后融合）
  - `Inception.py`：InceptionTime（对标签做 `log` 变换）
  - `plsr.py`：PLSR（统计特征：均值/标准差/最大值/RMS）
  - `XGBoost.py`：XGBoost（同 `plsr.py` 的统计特征）

## 数据约定（与代码保持一致）

### `data1/`（传统模型）

- 文件名：`{k}_{c}_p.csv`，例如 `0005_0010_p.csv`
- 标签解析：`k = float(k_str)/1000.0`、`c = float(c_str)/1000.0`
- CSV 列：第 1 列为 `time`，其后为肌肉通道；脚本会跳过 `time`，对其余通道提取统计特征作为输入

### `data2/`（深度模型）

- 文件名：`{k}_{c}_{F|D|P}_p.csv`（三文件为同一组样本的不同自由度）
- 脚本以 `Flexion_p/` 内的 `*_F_p.csv` 为“基准”，并在 `Deviation_p/`、`Pronation_p/` 查找同名对应样本组合
- 输入通道：默认仅使用 `SELECTED_MUSCLES` 中列（缺失列会补 0）；序列长度 `SEQ_LEN=800`，不足补 0，超长截断
- 部分 CSV 可能包含 `pos_flex/pos_dev/pos_pro` 等列，脚本默认不作为输入通道

## 环境依赖

建议使用 Python 3.9+（仅供参考），主要依赖：

- `numpy` `pandas` `matplotlib`
- `scikit-learn` `joblib`
- `torch`（深度模型需要）
- `xgboost`（仅 `XGBoost.py` 需要）

## 运行方式

在项目根目录执行（确保相对路径 `./data1`、`./data2` 可用）：

```bash
python plsr.py
python XGBoost.py

python GRU_bidir.py
python LSTM_bidir.py
python ResNet.py
python ResNet_v2.py
python Inception.py
```

## 输出产物

- 终端打印训练过程与评估指标（RMSE/MAE/R² 等）
- 弹出结果可视化图窗（`matplotlib`）
- 保存模型到 `model/`（例如 `model/bigru_fusion_model.pth`、`model/resnet_input_scaler.pkl` 等，可能覆盖同名文件）

## 常见问题

- 找不到数据：请确认工作目录在项目根目录，且 `data1/`、`data2/` 目录存在。
- 运行很慢：深度模型默认可用 GPU（`cuda`）则自动使用，否则走 CPU；可尝试减小 `NUM_EPOCHS/EPOCHS` 或 `SEQ_LEN` 做快速验证。
- 结果不可复现：深度训练可能受随机性与硬件影响；`train_test_split(random_state=42)` 只能保证拆分一致，不能保证训练完全一致。

## 贡献

欢迎提 Issue / PR（建议同时附上：运行脚本、环境版本、关键日志、复现步骤）。

## 许可证

本仓库当前未声明开源许可证；如需开源/发布，请先补充 `LICENSE` 并在 README 中更新说明。
