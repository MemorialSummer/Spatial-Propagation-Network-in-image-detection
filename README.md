# SESRN - Spatially Embedded Spiking Reservoir Network

SESRN 是一个基于 PyTorch 的**空间嵌入式尖峰储层神经网络**框架，在三维网格空间中构建神经元网络，通过局部连接与小世界长程连接实现信息传递与动态演化。

## 项目特性

- **三维空间神经元**: 神经元分布在 `GRID_X × GRID_Y × GRID_Z` 的三维网格中，每个神经元具有唯一的空间坐标 `(x, y, z)`
- **小世界拓扑**: 每个神经元自动建立 26 邻域局部连接 + 随机长程跳跃连接，形成小世界网络结构
- **泄漏积分动力学**: 多时间步循环更新 (`TIME_STEPS`)，带泄漏因子 (`LEAK_ALPHA`) 和 tanh 激活
- **稳态约束**: 通过 `torch.clamp` 限制神经元激活范围 `[-1.0, 1.0]`
- **端到端可训练**: 输入投影层 + 空间储层动力学 + 线性读出层，全部参数可通过反向传播优化

## 项目结构

```
SESRN/
├── configs/
│   └── config.py             # 超参配置（设备/网格尺寸/连接/动力学/训练/读出）
├── data/cifar10/              # 数据集存储目录
├── models/
│   ├── neuron.py             # Neuron 类（nid, 坐标, value, bias, 连接列表）
│   ├── connectivity.py       # ConnectivityBuilder（局部邻域 + 小世界长程连接构建器）
│   ├── spatial_network.py    # SpatialNetwork(nn.Module)（输入投影 → 循环动力学 → 读出）
│   └── readout.py            # （待实现）
├── train/
│   ├── trainer.py            # Trainer 训练器
│   └── losses.py             # （待实现）
├── visualization/
│   ├── activation_map.py     # visualize_activation() 3D 散点图可视化神经元激活
│   ├── graph_vis.py          # （待实现）
│   └── feature_region.py     # （待实现）
├── experiments/
│   ├── train_cifar10.py      # CIFAR-10 训练入口脚本
│   ├── ablation_smallworld.py  # （待实现）小世界消融实验
│   └── ablation_homeostasis.py # （待实现）稳态机制消融实验
├── outputs/
│   ├── checkpoints/          # .pth 模型检查点
│   ├── figures/              # 可视化图表输出
│   └── logs/                 # 日志文件
├── requirements.txt          # Python 依赖
└── README.md
```

## 核心模块

### 1. 配置 — `configs/config.py`

集中管理所有超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `"cuda"` | 计算设备 |
| `GRID_X/Y/Z` | `10` | 三维网格尺寸 (共 1000 个神经元) |
| `LOCAL_RADIUS` | `1` | 局部连接半径 |
| `LONG_RANGE_CONNECTIONS` | `3` | 每个神经元的长程连接数 |
| `TIME_STEPS` | `8` | 动力学循环步数 |
| `LEAK_ALPHA` | `0.7` | 泄漏积分因子 |
| `BATCH_SIZE/EPOCHS/LR` | `32 / 50 / 1e-3` | 训练超参数 |
| `NUM_CLASSES` | `10` | 分类类别数 |

### 2. 连接性 — `models/connectivity.py`

`ConnectivityBuilder` 负责构建网络的边集：

- **局部连接**: 对每个神经元遍历 26 邻域 (`dx, dy, dz ∈ {-1, 0, 1}`)，跳过自身
- **长程连接**: 每个神经元随机建立 `LONG_RANGE_CONNECTIONS` 条全局跳跃连接
- **坐标索引转换**: `position_to_index(x, y, z)` 将三维坐标映射为一维神经元 ID

```python
from models.connectivity import ConnectivityBuilder

builder = ConnectivityBuilder(10, 10, 10)
edges = builder.build()  # List[(src_id, dst_id)]
print(f"Total edges: {len(edges)}")
```

### 3. 空间网络 — `models/spatial_network.py`

`SpatialNetwork(nn.Module)` 是核心模型类：

```
输入图像 → input_proj(展平→tanh) → [TIME_STEPS 循环] → readout → logits
                                    ├─ 加权聚合邻居激活
                                    ├─ 加偏置
                                    ├─ tanh 激活
                                    └─ leaky 更新: h = α·h_old + (1-α)·h_new
```

前向传播返回 `(logits, hidden_states)`，其中 hidden_states 可用于可视化。

### 4. 神经元 — `models/neuron.py`

轻量级 `Neuron` 类，包含：
- 属性: `nid`, `x/y/z`, `value`, `bias`, `in_connections`, `out_connections`

### 5. 可视化 — `visualization/activation_map.py`

将隐藏状态 reshape 为三维网格后绘制 3D 散点图：

```python
from visualization.activation_map import visualize_activation

# h: [batch_size, num_neurons]
visualize_activation(h[0])  # 取第一个样本的激活状态
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖: `torch`, `torchvision`, `numpy`, `matplotlib`, `networkx`, `scikit-learn`, `tqdm`

### 在 CIFAR-10 上训练

```bash
python experiments/train_cifar10.py
```

训练流程：
1. 自动下载 CIFAR-10 到 `./data/`
2. 构建 10×10×10 的空间神经网络 (1000 个神经元)
3. Adam 优化器训练 50 个 epoch
4. 每个 epoch 结束后保存检查点到 `outputs/checkpoints/epoch_N.pth`

### 使用自己的数据

```python
import torch
from configs.config import *
from models.spatial_network import SpatialNetwork

model = SpatialNetwork().to(DEVICE)
x = torch.randn(BATCH_SIZE, 3, 32, 32).to(DEVICE)
logits, h = model(x)  # logits: [B, 10], h: [B, 1000]
```

## 待开发功能

- [ ] 完整的 LIF/IZH 尖峰神经元模型
- [ ] STDP 局部学习规则
- [ ] `models/readout.py`: MLP / 全局池化读出策略
- [ ] `train/trainer.py` / `train/losses.py`: 完整训练器与自定义损失函数
- [ ] `experiments/ablation_smallworld.py`: 小世界参数消融实验
- [ ] `experiments/ablation_homeostasis.py`: 泄漏因子 / 稳态范围消融实验
- [ ] `visualization/graph_vis.py`: 网络拓扑可视化（度分布、聚类系数）
- [ ] `visualization/feature_region.py`: 特征选择性分析（PCA、区域特化热力图）

## 许可证

本项目仅供学习和研究使用。


## 5月25日更新
connectivity.py 中删除了边列表的日志输出，并将边列表保存为 PyTorch 张量格式以供后续使用。并将edge建立时设置为set(),避免重复建立。以及和对应的边的权重对等，保持其顺序，防止边-权重对应顺序错乱
spatial_network.py 中将输出神经元宽度写进config.py，变为可调节。将图像尺寸写进config.py，变为可调节。目前仅支持正方形。