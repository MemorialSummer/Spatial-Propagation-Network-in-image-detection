import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import *
from models.connectivity import ConnectivityBuilder


class SpatialNetwork(nn.Module):
    def __init__(self, save_edges=True):

        # 初始化的时候手动带上父类nn.Module的初始化
        super().__init__()

        # 神经元总数
        self.num_neurons = NUM_NEURONS

        # 构建神经元的连接
        builder = ConnectivityBuilder( GRID_X, GRID_Y, GRID_Z, save_edges=save_edges )

        # 构建连接列表
        self.edges = builder.build()
        self.max_layer = min(GRID_X, GRID_Y, GRID_Z) // 2
        # 几何中心
        self.center_x = GRID_X // 2
        self.center_y = GRID_Y // 2
        self.center_z = GRID_Z // 2

        # 改进1：使用多个输出神经元（中心区域的所有神经元）
        radius = OUTPUT_NEURONS_NUMBER  # 取中心区域的神经元
        self.output_neurons = []
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    x = self.center_x + dx
                    y = self.center_y + dy
                    z = self.center_z + dz
                    if 0 <= x < GRID_X and 0 <= y < GRID_Y and 0 <= z < GRID_Z:
                        idx = x * GRID_Y * GRID_Z + y * GRID_Z + z
                        self.output_neurons.append(idx)

        # 预计算边的索引（避免每次forward都枚举）
        self.register_buffer('src_indices', torch.tensor([src for src, _ in self.edges]))
        self.register_buffer('dst_indices', torch.tensor([dst for _, dst in self.edges]))

        # 初始化权重和偏置
        self.weights = nn.Parameter(torch.randn(len(self.edges)) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_neurons))
        
        # 改进3：增强的readout网络
        self.readout = nn.Sequential(
            nn.Linear(len(self.output_neurons), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES)
        )
        
        # self.layer_norm = nn.LayerNorm(self.num_neurons)

        # 可学习卷积编码器：先提取特征，再插值到与 SPN 输入网格一致的尺寸
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )
        
        # 预计算网格信息
        self.grid_info = []
        self._input_neuron_indices_list = []
        self._precompute_grid_indices()
        self.register_buffer('input_neuron_indices', torch.tensor(self._input_neuron_indices_list, dtype=torch.long))

    def _precompute_grid_indices(self):
            """预计算所有网格的索引，避免在forward中重复计算"""
            for gx in range(GRID_X):
                for gy in range(GRID_Y):
                    # 预计算神经元索引
                    base_idx = gx * GRID_Y * GRID_Z + gy * GRID_Z
                    r_idx = base_idx + 0
                    g_idx = base_idx + 1
                    b_idx = base_idx + 2
                    
                    self._input_neuron_indices_list.extend([r_idx, g_idx, b_idx])
                    self.grid_info.append({
                        'gx': gx,
                        'gy': gy,
                        'r_idx': r_idx,
                        'g_idx': g_idx,
                        'b_idx': b_idx
                    })

    def _build_input_signal(self, input_images):
        """把输入图像映射为网络输入神经元的初始信号。"""
        batch_size = input_images.shape[0]
        input_signal = torch.zeros(
            batch_size,
            self.num_neurons,
            device=input_images.device,
            dtype=input_images.dtype
        )

        # 先通过可学习编码器提取特征，再插值到网格大小 (GRID_X, GRID_Y)
        encoded = self.encoder(input_images)
        encoded = F.interpolate(encoded, size=(GRID_X, GRID_Y), mode='bilinear', align_corners=False)

        for info in self.grid_info:
            gx = info['gx']
            gy = info['gy']

            r_value = encoded[:, 0, gx, gy]
            g_value = encoded[:, 1, gx, gy]
            b_value = encoded[:, 2, gx, gy]

            input_signal[:, info['r_idx']] = r_value
            input_signal[:, info['g_idx']] = g_value
            input_signal[:, info['b_idx']] = b_value

        return input_signal

    def forward(self, input_images):
        batch_size = input_images.shape[0]

        # 将图像信息映射为输入神经元信号
        input_signal = self._build_input_signal(input_images)

        # 步骤1：初始化所有节点为0
        h = torch.zeros(batch_size, self.num_neurons, device=input_images.device, dtype=input_images.dtype)

        # 确保索引在正确的设备上
        src_indices = self.src_indices.to(input_images.device)
        dst_indices = self.dst_indices.to(input_images.device)

        # 步骤3：优化信息传播 - 使用批量操作替代逐边循环
        for _ in range(TIME_STEPS):
            # 在每个 timestep 都把输入神经元刷新为当前图像信号
            h[:, self.input_neuron_indices] = input_signal[:, self.input_neuron_indices]

            # 初始激活
            h = torch.tanh(h)

            # 批量计算所有边的贡献
            # 一次性计算所有源节点的加权值
            src_values = h[:, src_indices]  # [batch, num_edges]
            weighted_values = src_values * self.weights  # [batch, num_edges]
            
            # 批量累加到目标节点（使用index_add_，GPU加速）
            new_h = torch.zeros_like(h)
            new_h.index_add_(1, dst_indices, weighted_values)
            
            # 添加偏置
            new_h += self.bias
            
            # 激活
            new_h = torch.tanh(new_h)
            
            # Leaky recurrent update
            h = LEAK_ALPHA * h + (1 - LEAK_ALPHA) * new_h
            
            # Homeostasis
            h = torch.clamp(h, -1.0, 1.0)


        output_values = h[:, self.output_neurons]  # [batch, num_output_neurons]
        out = self.readout(output_values)
        # print(output_values.abs().mean())

        return out, h