import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts#ReduceLROnPlateau
from torch.utils.data import DataLoader

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
os.makedirs('outputs/checkpoints', exist_ok=True)
os.makedirs('outputs/edges', exist_ok=True)

# ImageNet的均值方差（固定值）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 训练集数据增强
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0), ratio=(3/4, 4/3)),  # 随机裁剪缩放
    transforms.RandomHorizontalFlip(),               # 随机水平翻转，增加镜像不变性
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色扰动
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)     # 归一化到0附近
])

# 验证集数据增强
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


trainset = torchvision.datasets.ImageNet(
    root=IMAGENET_ROOT,
    split='train',
    transform=transform_train
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

valset = torchvision.datasets.ImageNet(
    root=IMAGENET_ROOT,
    split='val',
    transform=transform_val
)

valloader = DataLoader(
    valset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = torch.device(DEVICE)
print(f"Using device: {device}")
model = SpatialNetwork().to(device)
# edge_index = torch.load(edge_path, map_location=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
trainer = Trainer(model, optimizer, criterion, device, scheduler)

best_val_acc = 0
best_epoch = 0

history = {
    'epoch': [],
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

for epoch in range(EPOCHS):
    # 训练一个epoch
    train_loss, train_acc = trainer.train_epoch(trainloader)
    
    # 验证
    val_loss, val_acc = trainer.validate(valloader)
    
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录数据
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train: loss={train_loss:.4f} acc={train_acc:.2f}%")
    print(f"  Val:   loss={val_loss:.4f} acc={val_acc:.2f}%")
    print(f"  LR: {current_lr:.6f}")
    
    # 保存最佳模型（基于验证集）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "outputs/best_model.pth")
        print(f"  -> 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
    

    torch.save(model.state_dict(), f"outputs/checkpoints/epoch_{epoch+1}.pth")
    
    # 保存历史记录
    with open('outputs/logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")

# ========== 新增：训练结束后绘制折线图 ==========
def plot_training_history(history, save_path='outputs/figures/training_curves.png'):
    """绘制训练曲线（包含验证集）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history['epoch']
    
    # 图1：训练和验证Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curves', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图2：训练和验证准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Accuracy Curves', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3：学习率变化
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2, marker='^', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 图4：训练和验证的差距（过拟合程度）
    axes[1, 1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    axes[1, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    axes[1, 1].fill_between(epochs, history['train_acc'], history['val_acc'], 
                            alpha=0.3, color='gray', label='Gap')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Overfitting Analysis (Gap = Train - Val)', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved to {save_path}")

# 绘制图表
plot_training_history(history)

# 打印最佳结果
best_epoch = history['val_acc'].index(max(history['val_acc']))
print(f"\n{'='*50}")
print(f"Best Result: Epoch {best_epoch+1}")
print(f"Best Accuracy: {history['val_acc'][best_epoch]:.2f}%")
print(f"Corresponding Loss: {history['val_loss'][best_epoch]:.4f}")
print(f"{'='*50}")
