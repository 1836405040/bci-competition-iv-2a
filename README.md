# BCI Competition IV 2a：运动想象脑电四分类

这是一个面向公开基准数据集 **BCI Competition IV dataset 2a** 的可复现实验项目。任务是区分左手、右手、双脚和舌头四类运动想象，包含 9 名受试者、22 个 EEG 通道。代码提供：

- 4--38 Hz 带通滤波、保留 EEG 通道、按事件切分 0.5--2.5 s epoch，并让 MNE 按标注拒绝明显坏段；
- CSP 特征 + shrinkage LDA 与线性 SVM 两个经典基线；
- `src/bci_2a/models.py` 中的轻量 EEGNet 对照模型定义；
- 受试者独立的 5 折 Stratified CV。标准化和 CSP 均在每个训练折内拟合，避免数据泄漏；
- 每个受试者的结果、元数据和跨受试者汇总 CSV。

## 安装

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## 下载数据并运行经典模型

数据文件来自 BCI Competition IV 官方公开归档，命名为 `A01T.gdf` 到 `A09T.gdf`。第一次运行会缓存约 440 MB 的 `BCICIV_2a_gdf.zip`，之后按需解压；数据不提交到 Git：

```bash
python -m bci_2a.experiment --download --data-root data --output results
```

默认同时评估 `csp-lda`、`csp-svm` 和 `eegnet`。EEGNet 每个折从头训练；可用 `--eegnet-epochs 100` 调整训练轮数。

也可以只跑部分受试者或单个模型：

```bash
python -m bci_2a.experiment --subjects 1 2 3 --models csp-lda --download
```

输出：

- `results/subject_results.csv`：每个受试者、模型的 5 折平均准确率、标准差和平衡准确率；
- `results/summary.csv`：跨 9 名受试者的均值和受试者间标准差；
- `results/subject_XX_metadata.json`：采样率、通道名和有效 trial 数。

## EEGNet

`EEGNet` 已实现为 PyTorch 模块，并已接入上述 CLI，输入形状为 `(batch, 1, channels, samples)`，输出 4 类 logits。训练和评估在服务器上完成，原始数据不随仓库分发。

## 实测结果

2026-09-16 在 RTX 5090 服务器上以随机种子 42 完成 9 名受试者、受试者内分层 5 折交叉验证：

| 模型 | 9 人平均准确率 | 受试者间标准差 |
|---|---:|---:|
| CSP + shrinkage LDA | 61.54% | 17.66% |
| CSP + linear SVM | **62.28%** | 17.83% |
| EEGNet（100 epochs） | 52.31% | 19.01% |

随机水平为 25%。该结果衡量同一受试者、同一训练 session 内的新 trial 泛化，不代表跨 session 或跨受试者泛化。完整逐受试者结果与命令见 [`reports/RESULTS.md`](reports/RESULTS.md)。

## 数据与引用

请遵守数据集发布方的使用条款，并在论文/报告中引用：

> Tangermann M, et al. Review of the BCI Competition IV. *Frontiers in Neuroscience*, 2012, 6:55. DOI: 10.3389/fnins.2012.00055.

数据集说明：BCI Competition IV 2a（9 subjects, 22 EEG channels, four motor-imagery classes）。数据不随仓库分发。

## 复现注意事项

- 本项目默认只使用训练 session（`AxxT.gdf`）；官方评测 session（`AxxE.gdf`）没有标签，不应参与训练折。
- 交叉验证单位是 trial，模型按受试者分别训练和评估；如果要测试跨受试者泛化，应另行使用 GroupKFold，并在报告中明确任务定义。
- 滤波和 epoch 参数在 `load_subject` 中集中定义，随机种子默认 42。
- 数据下载、依赖版本和硬件差异可能造成小幅结果波动。
