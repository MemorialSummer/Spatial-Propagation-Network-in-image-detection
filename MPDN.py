class NeuroCore:
    '''
     生成单个神经元 NeuroCore类，包含神经元的基本属性和方法
     基本属性：
     神经元ID nid
     神经元在网络中的位置坐标 x,y,z
     输入神经元列表 input_cores
     输出神经元列表 output_cores
     权重列表 weights
     偏置列表 biases
     基本方法：
     新增输入神经元链接 new_input_neuro_link
     新增输出神经元链接 new_output_neuro_link
     删除输入神经元链接 del_input_neuro_link
     删除输出神经元链接 del_output_neuro_link
     修改权重 change_weight
     修改偏置 change_bias
     获取神经元位置 get_position
     获取神经元ID get_id
     获取输入神经元列表 get_input_cores
     获取输出神经元列表 get_output_cores
     获取权重列表 get_weights
     获取偏置列表 get_biases
     获取神经元信息 get_core_info
    '''
    # 初始化神经元
    def __init__(self, nid: int, x: int, y: int, z: int):
        self.nid = nid
        self.x = x
        self.y = y
        self.z = z
        self.input_cores = []
        self.output_cores = []
        self.weights = []
        self.biases = []

    # 新增输入神经元链接
    def new_input_neuro_link(self, core_id):
        if core_id not in self.input_cores:
            self.input_cores.append(core_id)
        return len(self.input_cores) - 1
    
    # 新增输出神经元链接
    def new_output_neuro_link(self, core_id):
        if core_id not in self.output_cores:
            self.output_cores.append(core_id)
        return len(self.output_cores) - 1
    
    # 删除输入神经元链接
    def del_input_neuro_link(self, core_id):
        if core_id in self.input_cores:
            self.input_cores.remove(core_id)
    
    # 删除输出神经元链接
    def del_output_neuro_link(self, core_id):
        if core_id in self.output_cores:
            self.output_cores.remove(core_id)
    
    # 修改权重
    def change_weight(self, weights):
        self.weights = weights

    # 修改神经元ID
    def change_id(self, nid):
        self.nid = nid

    # 修改神经元位置
    def change_positions(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # 修改偏置
    def change_bias(self, biases):
        self.biases = biases

    # 获取神经元位置
    def get_position(self):
        return (self.x, self.y, self.z)
    
    # 获取神经元ID
    def get_id(self):
        return self.nid
    
    # 获取输入神经元列表
    def get_input_cores(self):
        return self.input_cores
    
    # 获取输出神经元列表
    def get_output_cores(self):
        return self.output_cores
    
    # 获取权重列表
    def get_weights(self):
        return self.weights
    
    # 获取偏置列表
    def get_biases(self):
        return self.biases

    # 获取神经元信息
    def get_core_info(self):
        return {
            "id": self.nid,
            "position": (self.x, self.y, self.z),
            "input_cores": self.input_cores,
            "output_cores": self.output_cores,
            "weights": self.weights,
            "biases": self.biases
        }


class NeuroCoreNetwork:
    # 神经元网络配置类，包含多个神经元 NeuroCore 对象
    def __init__(self, Rows: int, Cols: int, Height: int):
        self.input_cores = []
        self.output_cores = []
        self.cores = []
        self.network_Rows = Rows
        self.network_Cols = Cols
        self.network_Height = Height

    # 自动初始化输入输出神经元
    def self_init_io(self,input_cores: list | None = None,output_cores: list | None = None):
        # 计算立方体边长 N
        N = int(len(self.cores) ** (1/3))
        # ---------- Input cores ----------
        if input_cores is None:
            auto_inputs = []
            for z in range(N):          # 上 → 下
                for x in range(N):      # 左 → 右
                    for y in range(N):  # 前 → 后
                        if ( x == 0 or x == N - 1 or y == 0 or y == N - 1 or z == 0 or z == N - 1 ):
                            auto_inputs.append((x, y, z))
            self.input_cores = auto_inputs
        else:
            self.input_cores = input_cores

        # ---------- Output cores ----------
        if output_cores is None:
            center = (N // 2, N // 2, N // 2)
            self.output_cores = [center]
        else:
            self.output_cores = output_cores

    # 批量初始化神经元
    def batch_initialize_cores(self, X: int, Y: int, Z: int):
        self.network_Rows = X
        self.network_Cols = Y
        self.network_Height = Z
        core_id = 0
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    core = NeuroCore(core_id, x, y, z)
                    self.cores.append(core)
                    core_id += 1
    
    # 获取神经元对象
    def get_core(self, core_id: int):
        for core in self.cores:
            if core.get_id() == core_id:
                return core
        return None

    # 添加神经元对象
    def add_core(self, core: NeuroCore):
        core_id = core.get_id()
        if self.get_core(core_id) is not None:
            print(f"Core with ID {core_id} already exists.（神经元ID已存在。）")
            while self.get_core(core_id) is not None:
                core_id += 1
            core.change_id(core_id)
            print(f"New Core ID（新神经元ID）: {core_id}")
        self.cores.append(core)

    # 删除神经元对象
    def remove_core(self, core_id: int):
        core = self.get_core(core_id)
        if core is not None:
            self.cores.remove(core)

# def train_model(Network: NeuroCoreNetwork, epochs: int, batch_size: int, learning_rate: float, data: list, data_route: str):
    

config = NeuroCoreNetwork(10, 10, 5)
config.add_core(NeuroCore(1, 0, 0, 0))
config.add_core(NeuroCore(2, 1, 0, 0))
config.get_core(2).new_input_neuro_link(1)
print(config.get_core(2).get_input_cores())
