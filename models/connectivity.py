import random
from configs.config import LOCAL_RADIUS,LONG_RANGE_CONNECTIONS
import torch
import os

def position_to_index(x, y, z, grid_y, grid_z):
    return x * grid_y * grid_z + y * grid_z + z


class ConnectivityBuilder:
    def __init__(self, grid_x, grid_y, grid_z, save_edges=True):
        '''
            初始化连接性构建器
            
            参数:
                grid_x: 网格x轴大小
                grid_y: 网格y轴大小
                grid_z: 网格z轴大小
                save_edges: 是否保存边信息
        '''
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.save_edges = save_edges

        # self.num_neurons = grid_x * grid_y * grid_z

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
        edges = set()

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
                            edges.add((src, dst))
                        
                        # 同层横向连接（50%概率）
                        elif dst_layer == src_layer:
                            if random.random() < 0.5:
                                edges.add((src, dst))
                    # small-world long range
                    count = 0
                    while count < LONG_RANGE_CONNECTIONS:
                        tx = random.randint(0, self.grid_x - 1)
                        ty = random.randint(0, self.grid_y - 1)
                        tz = random.randint(0, self.grid_z - 1)

                        dst = position_to_index(tx, ty, tz, self.grid_y, self.grid_z)
                        # with open("outputs/logs/connections.txt", 'a') as f:
                        #     f.write(f"{src},{dst}\n")
                        if src != dst and (src, dst) not in edges:
                            edges.add((src, dst))
                            count += 1
        edges = list(edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        i = 0
        while os.path.exists(f"./outputs/edges/edge_index_{i}.pt"):
            i += 1
        if self.save_edges:
            torch.save(edge_index, f"./outputs/edges/edge_index_{i}.pt")
        return edges