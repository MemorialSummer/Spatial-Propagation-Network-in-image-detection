import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts#ReduceLROnPlateau

import torchvision
import torchvision.transforms as transforms

from configs.config import *
from models.spatial_network import SpatialNetwork
from train.trainer import Trainer

# 新增导入
import matplotlib.pyplot as plt
import json
import os

# 启用cudnn自动调优
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False

# ========== 新增：创建保存目录 ==========
os.makedirs('outputs/logs', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])


trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,  # 增加worker数量
    pin_memory=True,  # 加速数据传输
    prefetch_factor=4,  # 预取更多批次
    persistent_workers=True  # 保持worker进程
)


device = torch.device(DEVICE)

model = SpatialNetwork().to(device)

# 使用更好的优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
trainer = Trainer(model, optimizer, criterion, device, scheduler)

# ========== 新增：记录训练历史 ==========
history = {
    'epoch': [],
    'loss': [],
    'acc': [],
    'lr': []
}

for epoch in range(EPOCHS):

    loss, acc = trainer.train_epoch(trainloader)
    
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录数据
    history['epoch'].append(epoch + 1)
    history['loss'].append(loss)
    history['acc'].append(acc)
    history['lr'].append(current_lr)


    print(f"Epoch {epoch+1} | loss={loss:.4f} acc={acc:.2f}% lr={current_lr:.6f}")

    torch.save(
        model.state_dict(),
        f"outputs/checkpoints/epoch_{epoch+1}.pth"
    )

    # ========== 新增：每个epoch结束后保存历史记录 ==========
    # 保存为JSON文件
    with open('outputs/logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # 保存为文本文件（方便查看）
    with open('outputs/logs/training_history.txt', 'w') as f:
        f.write("Epoch\tLoss\tAcc(%)\tLR\n")
        for i in range(len(history['epoch'])):
            f.write(f"{history['epoch'][i]}\t{history['loss'][i]:.4f}\t{history['acc'][i]:.2f}\t{history['lr'][i]:.6f}\n")


# ========== 新增：训练结束后绘制折线图 ==========
def plot_training_history(history, save_path='outputs/figures/training_curves.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = history['epoch']
    
    # 图1：Loss曲线
    axes[0, 0].plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图2：准确率曲线
    axes[0, 1].plot(epochs, history['acc'], 'r-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training Accuracy', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3：学习率变化曲线
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2, marker='^', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')  # 学习率用对数坐标
    
    # 图4：Loss和Acc双轴图
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Loss')
    line2 = ax2.plot(epochs, history['acc'], 'r-', linewidth=2, label='Accuracy')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='b', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', color='r', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Loss vs Accuracy', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved to {save_path}")

# 绘制图表
plot_training_history(history)

# 打印最佳结果
best_epoch = history['acc'].index(max(history['acc']))
print(f"\n{'='*50}")
print(f"Best Result: Epoch {best_epoch+1}")
print(f"Best Accuracy: {history['acc'][best_epoch]:.2f}%")
print(f"Corresponding Loss: {history['loss'][best_epoch]:.4f}")
print(f"{'='*50}")