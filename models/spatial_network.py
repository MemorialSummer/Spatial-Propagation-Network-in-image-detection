import torch
import torch.nn as nn
# import torch.nn.functional as F

from configs.config import *
from models.connectivity import ConnectivityBuilder


class SpatialNetwork(nn.Module):
    def __init__(self):

        # 初始化的时候手动带上父类nn.Module的初始化
        super().__init__()

        # 神经元总数
        self.num_neurons = NUM_NEURONS

        # 构建神经元的连接
        builder = ConnectivityBuilder( GRID_X, GRID_Y, GRID_Z )

        # 构建连接列表
        self.edges = builder.build()
        self.max_layer = min(GRID_X, GRID_Y, GRID_Z) // 2
        # 几何中心
        self.center_x = GRID_X // 2
        self.center_y = GRID_Y // 2
        self.center_z = GRID_Z // 2

        # 唯一输出神经元
        self.output_neurons = [self.center_x * GRID_Y * GRID_Z + self.center_y * GRID_Z + self.center_z]
        # 初始化权重和偏置
        self.weights = nn.Parameter(
            torch.randn(len(self.edges)) * 0.02
        )
        self.bias = nn.Parameter(
            torch.zeros(self.num_neurons)
        )

        # # 修改1：输入投影只映射到输入节点
        # # 假设输入图像是32x32x3，你需要计算实际输入维度
        # input_dim = 32 * 32 * 3  # 改成你实际的图像尺寸
        # self.input_proj = nn.Linear(
        #     input_dim,
        #     len(self.input_neurons)  # 只投影到输入节点的数量
        # )

        # Readout
        self.readout = nn.Linear(
            1, # 只读取输出节点的数量
            NUM_CLASSES
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # x = x.view(batch_size, -1)

        # 步骤1：初始化所有节点为0
        # 初始化整个神经系统
        h = torch.zeros(batch_size, self.num_neurons, device=x.device)

        # ==========================================
        # 将32x32图像压缩映射到10x10输入平面
        # ==========================================

        cell_h = 32 / GRID_X
        cell_w = 32 / GRID_Y

        for gx in range(GRID_X):
            for gy in range(GRID_Y):
                # 对应原图区域
                x_start = int(gx * cell_h)
                x_end = int((gx + 1) * cell_h)

                y_start = int(gy * cell_w)
                y_end = int((gy + 1) * cell_w)

                # 防止边界问题
                x_end = max(x_end, x_start + 1)
                y_end = max(y_end, y_start + 1)

                # ==========================
                # RGB局部平均
                # ==========================

                r = x[:, 0, x_start:x_end, y_start:y_end].mean(dim=(1, 2))
                g = x[:, 1, x_start:x_end, y_start:y_end].mean(dim=(1, 2))
                b = x[:, 2, x_start:x_end, y_start:y_end].mean(dim=(1, 2))

                # ==========================
                # 映射到三个输入平面
                # ==========================

                r_idx = (
                    gx * GRID_Y * GRID_Z
                    + gy * GRID_Z
                    + 0
                )

                g_idx = (
                    gx * GRID_Y * GRID_Z
                    + gy * GRID_Z
                    + 1
                )

                b_idx = (
                    gx * GRID_Y * GRID_Z
                    + gy * GRID_Z
                    + 2
                )

                h[:, r_idx] = r
                h[:, g_idx] = g
                h[:, b_idx] = b

        # 初始激活
        h = torch.tanh(h)
        # 步骤3：信息传播（从外向内）
        for _ in range(TIME_STEPS):
            new_h = torch.zeros_like(h)
            for edge_id, (src, dst) in enumerate(self.edges):
                new_h[:, dst] += (
                    h[:, src] * self.weights[edge_id]
                )

            new_h += self.bias

            new_h = torch.tanh(new_h)

            # Leaky recurrent update
            h = LEAK_ALPHA * h + (1 - LEAK_ALPHA) * new_h

            # Homeostasis
            h = torch.clamp(h, -1.0, 1.0)

        output_values = h[:, self.output_neurons]  # [batch, num_output_neurons]
        out = self.readout(output_values)

        return out, h