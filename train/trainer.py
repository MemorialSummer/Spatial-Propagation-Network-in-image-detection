# import torch
from tqdm import tqdm


class Trainer:
    def __init__( self, model, optimizer, criterion, device, scheduler=None):
        '''
        初始化训练器
        
        参数:
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            device: 设备 CPU 或 GPU
            scheduler: 学习率调度器
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
    def train_epoch(self, loader):
        '''
            训练一个完整的epoch
        '''
        self.model.train()

        total_loss = 0
        correct = 0
        # 已经训练的样本数
        total = 0
        
        # 将训练数据loader包装成tqdm进度条可视化训练进程。
        pbar = tqdm(loader)

        for images, labels in pbar:
            # 使用non_blocking异步传输
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # 每次手动清零梯度
            self.optimizer.zero_grad()

            # 前向传播，计算损失，反向传播，更新参数
            outputs, _ = self.model(images) 
            '''
                outputs形如：
                猫，狗
                [ 
                    [0.1, 0.9],
                    [0.8, 0.2]
                ]
            '''
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # 用于统计计算当前的损失和准确率，并更新进度条描述
            total_loss += loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            '''
                outputs.max(1)返回每行的最大值和对应的索引形如：
                (tensor([0.9, 0.2]), tensor([0, 1]))

                predicted形如：
                [1, 0]
            '''

            total += labels.size(0)

            # 训练到当前为止，预测正确的样本数，即准确率
            correct += predicted.eq(labels).sum().item()
            acc = 100. * correct / total

            pbar.set_description(
                f"loss={loss.item():.4f} acc={acc:.2f}%"
            )

        if self.scheduler:
            self.scheduler.step(total_loss / len(loader))

        return total_loss / len(loader), acc