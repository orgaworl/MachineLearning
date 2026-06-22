# 激活函数对比实验

基于 PyTorch（MPS 加速）的深度学习对比实验，在 MNIST 和 CIFAR-10 数据集上系统比较 ReLU、Leaky ReLU、Swish、GELU 四种激活函数，分析其对训练效果、死亡神经元比例和激活值分布的影响。

## 实验内容

- **激活函数**：ReLU、Leaky ReLU、Swish（SiLU）、GELU
- **数据集**：MNIST、CIFAR-10（标准）、CIFAR-10（轻量配置）
- **网络结构**：4 层卷积（32→64→128→256 通道）+ BatchNorm + 2 层全连接，激活函数可插拔
- **指标**：训练/验证 Acc & Loss、死亡神经元比例（动态 + 最终统计）、激活值分布快照

## 环境要求

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)（包管理）
- Apple Silicon Mac（MPS）或 CUDA GPU，CPU 也可运行

## 安装

```bash
git clone <repo>
cd ML
uv sync --dev
```

首次运行会下载 PyTorch（~500MB）、MNIST（~11MB）、CIFAR-10（~170MB）。

## 使用

### 运行全部实验

```bash
uv run python experiments/run_all.py
```

共 12 组实验（3 数据集 × 4 激活函数），支持断点续跑——中断后重新运行会自动跳过已完成的组。

预计时长（M4 Pro）：
- MNIST × 4：约 20 分钟
- CIFAR-10 × 4：约 60 分钟
- CIFAR-10 轻量 × 4：约 20 分钟

### 生成对比图

```bash
uv run python plots/plot_training_curves.py   # 训练曲线
uv run python plots/plot_dead_neurons.py      # 死亡神经元
uv run python plots/plot_activation_dist.py   # 激活值分布
```

图片输出到 `plots/output/`（300 dpi PNG）。

### 运行测试

```bash
uv run pytest tests/ -v
```

## 项目结构

```
ML/
├── pyproject.toml                     # 依赖配置
├── src/activation_exp/
│   ├── model.py                       # CNN 网络定义
│   ├── configs.py                     # 实验配置（12 组）
│   ├── data.py                        # 数据集加载
│   ├── metrics.py                     # 死亡神经元检测、激活分布采样
│   ├── trainer.py                     # 训练循环
│   └── utils.py                       # 设备选择、随机种子
├── experiments/
│   └── run_all.py                     # 实验入口
├── plots/
│   ├── plot_training_curves.py
│   ├── plot_dead_neurons.py
│   ├── plot_activation_dist.py
│   └── output/                        # 生成的图片
├── results/                           # 原始实验数据（JSON + npz）
│   └── {dataset}_{activation}/
│       ├── metrics.json               # 每 epoch 的 loss/acc/dead_ratio
│       └── activation_dist.npz        # 激活值分布快照
└── tests/
    ├── test_model.py
    ├── test_metrics.py
    └── test_trainer.py
```

## 输出文件说明

**`results/{dataset}_{activation}/metrics.json`**：

```json
{
  "config": {"dataset": "cifar10", "activation": "relu", "epochs": 30},
  "epoch":      [1, 2, ...],
  "train_loss": [...],
  "train_acc":  [...],
  "val_loss":   [...],
  "val_acc":    [...],
  "dead_ratio": [...]
}
```

**`results/{dataset}_{activation}/activation_dist.npz`**：每 5 epoch 采样一次第 2 卷积层后的激活值，key 为 `epoch_1`、`epoch_5`、`epoch_10` 等。

## 生成图片列表

| 文件 | 内容 |
|------|------|
| `training_curves_mnist.png` | MNIST 四种激活函数训练/验证曲线对比 |
| `training_curves_cifar10.png` | CIFAR-10 训练/验证曲线对比 |
| `training_curves_cifar10_lite.png` | CIFAR-10 轻量版训练/验证曲线对比 |
| `dead_neurons_dynamic.png` | 三个数据集上死亡神经元比例随训练变化折线图 |
| `dead_neurons_final.png` | 训练结束时各激活函数死亡神经元比例柱状图 |
| `activation_dist_mnist.png` | MNIST 各激活函数在不同训练阶段的激活值分布 |
| `activation_dist_cifar10.png` | CIFAR-10 激活值分布 |
| `activation_dist_cifar10_lite.png` | CIFAR-10 轻量版激活值分布 |
