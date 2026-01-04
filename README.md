# class NeuroCore:
## 生成单个神经元 NeuroCore类，包含神经元的基本属性和方法
### 基本属性
神经元ID nid
神经元在网络中的位置坐标 x,y,z
输入神经元列表 input_cores
输出神经元列表 output_cores
权重列表 weights
偏置列表 biases
### 基本方法
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