import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from configs.config import *
from models.spatial_network import SpatialNetwork

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def find_latest_checkpoint():
    """寻找最新checkpoint"""

    checkpoint_dir = "outputs/checkpoints"

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth")
    ]

    if len(checkpoints) == 0:
        return None

    checkpoints.sort(
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    return os.path.join(checkpoint_dir, checkpoints[-1])


def find_latest_edge_file():
    """寻找最新edge_index"""

    edge_dir = "outputs/edges"

    if not os.path.exists(edge_dir):
        return None

    edge_files = [
        f for f in os.listdir(edge_dir)
        if f.endswith(".pt")
    ]

    if len(edge_files) == 0:
        return None

    edge_files.sort(
        key=lambda x: int(
            x.split("_")[-1].split(".")[0]
        )
    )

    return os.path.join(edge_dir, edge_files[-1])


def load_model(checkpoint_path, edge_path, device):

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Loading edge index: {edge_path}")

    # =========================
    # 读取edge_index
    # shape: [2, num_edges]
    # =========================

    edge_index = torch.load(edge_path, map_location=device)

    src_indices = edge_index[0]
    dst_indices = edge_index[1]

    num_edges = edge_index.shape[1]

    print(f"Loaded edges: {num_edges}")

    # =========================
    # 创建模型
    # =========================

    model = SpatialNetwork().to(device)

    # =========================
    # 替换边
    # =========================

    model.src_indices = src_indices.to(device)
    model.dst_indices = dst_indices.to(device)

    # 非常关键：
    # weights长度必须匹配训练时
    model.weights = nn.Parameter(
        torch.randn(num_edges, device=device) * 0.02
    )

    # =========================
    # 加载checkpoint
    # =========================

    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    print("Model loaded successfully")

    return model


def evaluate(model, testloader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs, _ = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(testloader)

    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():

    device = torch.device(DEVICE)

    print(f"Using device: {device}")

    # =========================
    # 数据集
    # =========================

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Loading CIFAR10 test dataset...")

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Test samples: {len(testset)}")

    # =========================
    # loss
    # =========================

    criterion = nn.CrossEntropyLoss()

    # =========================
    # checkpoint
    # =========================

    checkpoint_path = find_latest_checkpoint()

    if checkpoint_path is None:
        print("No checkpoint found")
        return

    # =========================
    # edge index
    # =========================

    edge_path = find_latest_edge_file()

    if edge_path is None:
        print("No edge_index found")
        return

    # =========================
    # load model
    # =========================

    model = load_model(
        checkpoint_path,
        edge_path,
        device
    )

    # =========================
    # evaluate
    # =========================

    test_loss, test_acc = evaluate(
        model,
        testloader,
        criterion,
        device
    )

    print("\n" + "=" * 50)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()