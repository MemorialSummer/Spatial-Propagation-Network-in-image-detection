# MPDN - Multi-dimensional Neural Network

MPDN 是一个基于 Python 的多维神经网络模拟框架,用于构建和操作三维空间中的神经元网络。

## 项目特性

- **三维神经元网络**: 支持在三维空间坐标系中构建神经元网络
- **神经元管理**: 提供完整的神经元创建、删除、查询功能
- **链接管理**: 支持神经元之间的输入/输出链接建立与删除
- **权重与偏置**: 灵活的权重和偏置配置机制
- **自动初始化**: 支持自动配置输入/输出神经元位置

## 核心类

### NeuroCore (神经元类)

单个神经元的核心类,包含以下属性和方法:

#### 属性
- `nid`: 神经元唯一标识符
- `x, y, z`: 神经元在网络中的三维坐标位置
- `input_cores`: 输入神经元 ID 列表
- `output_cores`: 输出神经元 ID 列表
- `weights`: 权重列表
- `biases`: 偏置列表

#### 主要方法
- `new_input_neuro_link(core_id)`: 添加输入神经元链接
- `new_output_neuro_link(core_id)`: 添加输出神经元链接
- `del_input_neuro_link(core_id)`: 删除输入神经元链接
- `del_output_neuro_link(core_id)`: 删除输出神经元链接
- `change_weight(weights)`: 修改权重
- `change_bias(biases)`: 修改偏置
- `get_position()`: 获取神经元位置
- `get_id()`: 获取神经元 ID
- `get_core_info()`: 获取神经元完整信息

### NeuroCoreNetwork (神经网络管理类)

用于管理多个神经元对象的网络配置类。

#### 属性
- `cores`: 神经元对象列表
- `input_cores`: 输入神经元坐标列表
- `output_cores`: 输出神经元坐标列表
- `network_Rows, network_Cols, network_Height`: 网络三维尺寸

#### 主要方法
- `__init__(Rows, Cols, Height)`: 初始化网络尺寸
- `self_init_io(input_cores, output_cores)`: 自动初始化输入/输出神经元
  - 未指定时,自动将立方体表面神经元设为输入
  - 输出神经元默认为立方体中心点
- `batch_initialize_cores(X, Y, Z)`: 批量创建神经元
- `get_core(core_id)`: 根据ID获取神经元对象
- `add_core(core)`: 添加神经元对象
- `remove_core(core_id)`: 删除神经元对象

## 使用示例

### 创建单个神经元
```python
# 创建位于坐标 (0, 0, 0) 的神经元, ID 为 1
neuron = NeuroCore(nid=1, x=0, y=0, z=0)
```

### 创建神经元链接
```python
# 创建两个神经元
neuron1 = NeuroCore(1, 0, 0, 0)
neuron2 = NeuroCore(2, 1, 0, 0)

# 建立链接: neuron1 作为 neuron2 的输入
neuron2.new_input_neuro_link(1)
neuron1.new_output_neuro_link(2)

# 配置权重和偏置
neuron2.change_weight([0.5])
neuron2.change_bias([0.1])
```

### 构建网络
```python
# 初始化网络
network = NeuroCoreNetwork(Rows=10, Cols=10, Height=5)

# 批量创建 5x5x5 的神经元网络
network.batch_initialize_cores(X=5, Y=5, Z=5)

# 自动配置输入输出
network.self_init_io()

# 手动添加单个神经元
new_neuron = NeuroCore(100, 2, 2, 2)
network.add_core(new_neuron)

# 查询神经元
target_neuron = network.get_core(2)
if target_neuron:
    print(target_neuron.get_core_info())

# 删除神经元
network.remove_core(100)
```

### 完整示例
```python
# 创建网络
config = NeuroCoreNetwork(10, 10, 5)

# 添加神经元并建立链接
config.add_core(NeuroCore(1, 0, 0, 0))
config.add_core(NeuroCore(2, 1, 0, 0))

# 建立输入链接
config.get_core(2).new_input_neuro_link(1)

# 获取输入列表
print(config.get_core(2).get_input_cores())  # 输出: [1]
```

## 项目结构

```
MPDN/
├── MPDN.py           # 核心实现文件
├── test.py           # 测试文件
└── README.md         # 项目文档
```

## 待开发功能

- [ ] 神经网络前向传播实现
- [ ] 反向传播训练算法
- [ ] 数据加载与预处理模块
- [ ] 可视化工具
- [ ] 更多网络拓扑支持

## 许可证

本项目仅供学习和研究使用。
