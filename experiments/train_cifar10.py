import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts#ReduceLROnPlateau

import torchvision
import torchvision.transforms as transforms

from configs.config import *
from models.spatial_network import SpatialNetwork
from train.trainer import Trainer

# 启用cudnn自动调优
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False

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


for epoch in range(EPOCHS):

    loss, acc = trainer.train_epoch(trainloader)

    print(
        f"Epoch {epoch+1} | loss={loss:.4f} acc={acc:.2f}%"
    )

    torch.save(
        model.state_dict(),
        f"outputs/checkpoints/epoch_{epoch+1}.pth"
    )