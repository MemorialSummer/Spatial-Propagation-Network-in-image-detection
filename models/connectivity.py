import random
from configs.config import LOCAL_RADIUS,LONG_RANGE_CONNECTIONS
# import math
# import torch


def position_to_index(x, y, z, grid_y, grid_z):
    return x * grid_y * grid_z + y * grid_z + z


class ConnectivityBuilder:
    def __init__(self, grid_x, grid_y, grid_z):
        '''
            初始化连接性构建器
            
            参数:
                grid_x: 网格x轴大小
                grid_y: 网格y轴大小
                grid_z: 网格z轴大小
        '''
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z

        self.num_neurons = grid_x * grid_y * grid_z

    def _get_neighbors_within_radius(self, x, y, z):
        neighbors = []
        
        # 生成半径范围内的所有偏移
        for dx in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
            for dy in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
                for dz in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
                    # 排除自身
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx = x + dx
                    ny = y + dy
                    nz = z + dz
                    
                    # 检查边界
                    if (0 <= nx < self.grid_x and 
                        0 <= ny < self.grid_y and 
                        0 <= nz < self.grid_z):
                        neighbors.append((nx, ny, nz))
        
        return neighbors

    def build(self):
        edges = []

        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for z in range(self.grid_z):

                    src = position_to_index(x, y, z, self.grid_y, self.grid_z)
                    src_layer = min(x, y, z, self.grid_x - 1 - x, self.grid_y - 1 - y, self.grid_z - 1 - z)
                    neighbors = self._get_neighbors_within_radius(x, y, z)
                    for nx, ny, nz in neighbors:
                        dst = position_to_index(nx, ny, nz, self.grid_y, self.grid_z)
                        dst_layer = min(nx, ny, nz, self.grid_x - 1 - nx, self.grid_y - 1 - ny, self.grid_z - 1 - nz)
                        
                        # 向内连接（从外层到内层）
                        if dst_layer > src_layer:
                            edges.append((src, dst))
                        
                        # 同层横向连接（50%概率）
                        elif dst_layer == src_layer:
                            if random.random() < 0.5:
                                edges.append((src, dst))
                    # small-world long range
                    for _ in range(LONG_RANGE_CONNECTIONS):
                        tx = random.randint(0, self.grid_x - 1)
                        ty = random.randint(0, self.grid_y - 1)
                        tz = random.randint(0, self.grid_z - 1)

                        dst = position_to_index(tx, ty, tz, self.grid_y, self.grid_z)
                        with open("outputs/logs/connections.txt", 'a') as f:
                            f.write(f"{src},{dst}\n")
                        if src != dst:
                            edges.append((src, dst))

        return edges