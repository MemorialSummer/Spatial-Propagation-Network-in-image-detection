import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
import glob

# 启用cudnn自动调优
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False

# ========== 创建保存目录 ==========
os.makedirs('outputs/logs', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/checkpoints', exist_ok=True)
os.makedirs('outputs/edges', exist_ok=True)

# ImageNet的均值方差（固定值）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 训练集数据增强
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# 验证集数据增强
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ========== 数据加载 ==========
trainset = torchvision.datasets.ImageNet(
    root=IMAGENET_ROOT,
    split='train',
    transform=transform_train
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=False,
    prefetch_factor=4,
    persistent_workers=True
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
    pin_memory=False
)

# ========== 设备 ==========
device = torch.device(DEVICE)
print(f"Using device: {device}")

# ========== 模型 ==========
model = SpatialNetwork().to(device)
# edge_index = torch.load(edge_path, map_location=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
trainer = Trainer(model, optimizer, criterion, device, scheduler)

# ========== 新增：查找最新checkpoint ==========
def find_latest_checkpoint():
    """查找outputs/checkpoints/下最新的epoch checkpoint"""
    checkpoint_dir = "outputs/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # 查找所有 epoch_*.pth 文件
    pattern = os.path.join(checkpoint_dir, "epoch_*.pth")
    files = glob.glob(pattern)
    
    if len(files) == 0:
        return None, 0
    
    # 提取epoch数字并排序
    epoch_nums = []
    for f in files:
        try:
            # 从 "epoch_123.pth" 提取 123
            num = int(os.path.basename(f).split('_')[1].split('.')[0])
            epoch_nums.append((num, f))
        except:
            continue
    
    if len(epoch_nums) == 0:
        return None, 0
    
    # 按epoch数字降序排列，取最新的
    epoch_nums.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_path = epoch_nums[0]
    
    return latest_path, latest_epoch

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """加载完整的checkpoint，返回起始epoch和最佳准确率"""
    print(f"\n🔄 发现checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ========== 兼容两种格式 ==========
    # 格式1（新）：{'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
    # 格式2（旧）：直接是 model.state_dict()
    
    if 'model_state_dict' in checkpoint:
        # 新格式：完整checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        best_epoch = checkpoint.get('best_epoch', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        history = checkpoint.get('history', {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        })
        
        print(f"  ✅ 从第 {start_epoch} 轮继续训练（新格式）")
        
    else:
        # 旧格式：只有模型权重
        print(f"  ⚠️ 检测到旧格式checkpoint（仅模型权重）")
        model.load_state_dict(checkpoint)
        
        # 优化器和调度器重新初始化（无法恢复状态）
        start_epoch = 0
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        print(f"  🔄 优化器和调度器将重新初始化")
        print(f"  📌 建议：训练1轮后会自动保存新格式checkpoint")
    
    print(f"  📊 历史最佳准确率: {best_val_acc:.2f}% (第 {best_epoch} 轮)")
    print(f"  📈 已有训练记录: {len(history['epoch'])} 轮")
    print(f"  🔢 早停计数器: {patience_counter}")
    
    return start_epoch, best_val_acc, best_epoch, patience_counter, history

# ========== 新增：保存完整checkpoint ==========
def save_full_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, 
                         best_epoch, patience_counter, history, train_loss, 
                         train_acc, val_loss, val_acc):
    """保存完整的checkpoint（包含所有状态）"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
        'history': history,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    
    checkpoint_path = f"outputs/checkpoints/epoch_{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

# ========== 核心：检查并加载checkpoint ==========
checkpoint_path, latest_epoch = find_latest_checkpoint()

if checkpoint_path is not None:
    start_epoch, best_val_acc, best_epoch, patience_counter, history = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, device
    )
else:
    print("\n🚀 开始全新训练")
    start_epoch = 0
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

# ========== 训练循环 ==========
print(f"\n{'='*60}")
print(f"训练轮次: {start_epoch + 1} 到 {EPOCHS}")
print(f"{'='*60}\n")

for epoch in range(start_epoch, EPOCHS):
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
    
    torch.save(model.state_dict(), f"outputs/checkpoints/epoch_{epoch+1}.pth")
    # 保存最佳模型（基于验证集）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "outputs/best_model.pth")
        print(f"  ✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        print(f"  ⏳ 早停计数器: {patience_counter}")
    
    # ========== 修改：保存完整的checkpoint（包含所有状态）==========
    save_full_checkpoint(
        model, optimizer, scheduler, epoch, best_val_acc, 
        best_epoch, patience_counter, history, train_loss, 
        train_acc, val_loss, val_acc
    )
    
    # 保存历史记录（JSON，方便查看）
    with open('outputs/logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    


# ========== 训练结束 ==========
print(f"\n{'='*60}")
print(f"训练完成！")
print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"{'='*60}")

# ========== 绘制训练曲线 ==========
def plot_training_history(history, save_path='outputs/figures/training_curves.png'):
    """绘制训练曲线（包含验证集）"""
    if len(history['epoch']) == 0:
        print("⚠️ 没有训练数据可绘制")
        return
    
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
    plt.close()  # 改为close，避免在无GUI环境报错
    print(f"📊 训练曲线已保存到: {save_path}")

# 绘制图表
plot_training_history(history)

# 打印最佳结果
best_idx = history['val_acc'].index(max(history['val_acc'])) if history['val_acc'] else 0
print(f"\n{'='*50}")
print(f"Best Result: Epoch {history['epoch'][best_idx]}")
print(f"Best Accuracy: {history['val_acc'][best_idx]:.2f}%")
print(f"Corresponding Loss: {history['val_loss'][best_idx]:.4f}")
print(f"{'='*50}")