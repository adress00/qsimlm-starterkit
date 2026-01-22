# qsimlm-starterkit：量子电路模拟 AI 入门套件

## 项目简介
这是一个为**量子电路模拟生成式模型**（Generative Models for Quantum Circuit Simulation）课程项目设计的**开箱即用入门套件**。
本项目旨在复现 Zhou et al. (2025) 论文中提到的**2量子比特特殊电路**实验设置，帮助初学者理解如何使用现代深度学习模型（如 Transformer/LSTM）来模拟和预测量子系统的状态。

## 核心功能
本套件包含完整的机器学习流水线：
1. **数据生成**: 自动生成包含随机旋转角度的量子电路，并使用 Qiskit 计算其真实的量子态矢量 (Statevector) 作为标签。
2. **AI 模型**:
   - **MLP (多层感知机)**: 一个简单的非自回归基线模型，用于快速验证。
   - **LSTM (长短期记忆网络)**: 一个基于 RNN 的非自回归基线模型，用于对比序列建模能力。
   - **Autoregressive (自回归模型)**: 结合了 LSTM 编码器和 Transformer 解码器的 Seq2Seq 架构，模拟大语言模型 (LLM) 的生成方式逐步预测量子态。
3. **评估系统**: 使用**量子态保真度 (State Fidelity)** 作为核心指标，衡量模型预测结果与真实物理状态的接近程度。

## 快速开始 (Quick Start)

### 1. 环境准备
本项目兼容 Linux, macOS 和 Windows。你需要安装 Python (推荐 3.9 或更高版本)。

**推荐方式：使用 Conda**
```bash
# 创建并激活环境
conda create -n qsimlm python=3.11
conda activate qsimlm

# 安装依赖
pip install -r requirements.txt
```

### 2. 一键运行测试
我们提供了自动化脚本，会自动检查环境、安装依赖并运行一个微型的训练演示。

- **Linux / macOS**:
  ```bash
  bash run.sh
  ```
- **Windows (PowerShell)**:
  ```powershell
  ./run.ps1
  ```

---

## 详细使用指南 (Manual Run)

如果你想深入了解或修改训练参数，可以通过命令行直接调用 Python 脚本。

### 核心训练命令
脚本入口为 `qsimlm.train_2q_special`模块。

#### 1. 运行 MLP 基准模型
适合初次尝试，速度极快。
```bash
python -m qsimlm.train_2q_special --model mlp --n_train 20000 --n_test 2000 --epochs 8
```

#### 2. 运行 Autoregressive (自回归) 模型
这是本项目的核心模型，展示了序列生成能力。
```bash
python -m qsimlm.train_2q_special --model autoreg --n_train 20000 --n_test 2000 --epochs 8
```

#### 3. 运行 LSTM 基准模型
介于 MLP 和 Transformer 之间的另一种基线。
```bash
python -m qsimlm.train_2q_special --model lstm --n_train 20000 --n_test 2000 --epochs 8
```

### 常用参数说明
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--model` | `autoreg` | 选择模型架构：`mlp`, `lstm` 或 `autoreg` |
| `--n_train` | `20000` | 训练集的样本数量。数据越多，模型泛化能力越强。 |
| `--n_test` | `2000` | 测试集的样本数量，用于计算最终的保真度。 |
| `--epochs` | `8` | 训练轮数。增加轮数通常能提升效果，但需注意过拟合。 |
| `--lr` | `3e-4` | 学习率 (Learning Rate)。 |
| `--seed` | `0` | 随机种子，保证结果可复现。 |
| `--batch_size` | `256` | 批次大小。 |
| `--noisy` | `False` | 开启噪声模拟 (使用密度矩阵作为标签)。 |
| `--noise_prob` | `0.01` | Depolarizing channel 的噪声强度 (仅在开启 `--noisy` 时有效)。 |
| `--use_trig` | `False` | 特征工程：将输入角度 $x$ 扩展为 $[\sin(x), \cos(x)]$。 |

## 项目文件结构

```text
qsimlm-starterkit/
├── qsimlm/
│   ├── data.py               # [核心] 数据生成器：构建量子电路并计算 Statevector
│   ├── models.py             # [核心] 模型库：定义 MLP 和 Seq2SeqAutoreg 模型结构
│   ├── train_2q_special.py   # [入口] 训练主脚本：包含训练循环和验证逻辑
│   ├── metrics.py            # 工具：计算量子态保真度 (Fidelity)
│   └── eval_2q_special.py    # 工具：独立评估脚本
├── requirements.txt          # 项目依赖列表
├── run.sh                    # Linux/Mac 的一键启动脚本
├── run.ps1                   # Windows 的一键启动脚本
└── README.md                 # 说明文档
```

## 进阶与挑战 (Optional)
如果你已经跑通了基础流程，可以尝试以下挑战任务（Ref: 论文进阶内容）：
1. **扩大规模**: 尝试修改 `data.py`，增加量子比特数量，观察模型训练难度的变化。
2. **混合态模拟**: 目前模型预测的是纯态 (Statevector)，尝试改用密度矩阵 (Density Matrix) 作为训练目标。
3. **物理约束优化**: 模型输出的数值可能不完全满足量子力学的物理约束（如归一化），尝试使用 `cvxpy` 添加后处理步骤来修正预测结果。

## 参考资料
- **引用**: Zhou et al., *Application of large language models to quantum state simulation*, Science China Physics, Mechanics & Astronomy (2025).
- 这是该论文实验设置的一个简化复现版本，专注于 **2-qubit Special Circuit** 场景。
