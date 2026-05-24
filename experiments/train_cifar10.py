import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from configs.config import *
from models.spatial_network import SpatialNetwork
from train.trainer import Trainer


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
    shuffle=True
)


device = torch.device(DEVICE)

model = SpatialNetwork().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

trainer = Trainer(
    model,
    optimizer,
    criterion,
    device
)


for epoch in range(EPOCHS):

    loss, acc = trainer.train_epoch(trainloader)

    print(
        f"Epoch {epoch+1} | loss={loss:.4f} acc={acc:.2f}%"
    )

    torch.save(
        model.state_dict(),
        f"outputs/checkpoints/epoch_{epoch+1}.pth"
    )