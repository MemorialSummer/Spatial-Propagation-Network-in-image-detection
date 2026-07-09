import json
import matplotlib.pyplot as plt
import numpy as np

# 读取两个json文件
with open('visualization/training_history_15*15*15.json', 'r') as f:
    data_r1 = json.load(f)

with open('visualization/training_history_26*26*26.json', 'r') as f:
    data_r3 = json.load(f)

# 创建图形，包含4个子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Comparison: 15*15*15 vs 26*26*26', fontsize=16, fontweight='bold')

# 1. Train Loss
ax1 = axes[0, 0]
ax1.plot(data_r1['epoch'], data_r1['train_loss'], label='15*15*15', color='blue', alpha=0.8)
ax1.plot(data_r3['epoch'], data_r3['train_loss'], label='26*26*26', color='red', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss')
ax1.set_title('Train Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Train Accuracy
ax2 = axes[0, 1]
ax2.plot(data_r1['epoch'], data_r1['train_acc'], label='15*15*15', color='blue', alpha=0.8)
ax2.plot(data_r3['epoch'], data_r3['train_acc'], label='26*26*26', color='red', alpha=0.8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Train Accuracy (%)')
ax2.set_title('Train Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Val Loss
ax3 = axes[1, 0]
ax3.plot(data_r1['epoch'], data_r1['val_loss'], label='15*15*15', color='blue', alpha=0.8)
ax3.plot(data_r3['epoch'], data_r3['val_loss'], label='26*26*26', color='red', alpha=0.8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Val Loss')
ax3.set_title('Validation Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Val Accuracy
ax4 = axes[1, 1]
ax4.plot(data_r1['epoch'], data_r1['val_acc'], label='15*15*15', color='blue', alpha=0.8)
ax4.plot(data_r3['epoch'], data_r3['val_acc'], label='26*26*26', color='red', alpha=0.8)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Val Accuracy (%)')
ax4.set_title('Validation Accuracy')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印最终结果对比
print("=" * 60)
print("FINAL RESULTS COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'15*15*15':<15} {'26*26*26':<15}")
print("-" * 60)
print(f"{'Final Train Loss':<25} {data_r1['train_loss'][-1]:<15.4f} {data_r3['train_loss'][-1]:<15.4f}")
print(f"{'Final Train Accuracy':<25} {data_r1['train_acc'][-1]:<15.2f}% {data_r3['train_acc'][-1]:<15.2f}%")
print(f"{'Final Val Loss':<25} {data_r1['val_loss'][-1]:<15.4f} {data_r3['val_loss'][-1]:<15.4f}")
print(f"{'Final Val Accuracy':<25} {data_r1['val_acc'][-1]:<15.2f}% {data_r3['val_acc'][-1]:<15.2f}%")
print("=" * 60)

# 找到最佳验证准确率
best_epoch_r1 = np.argmax(data_r1['val_acc'])
best_epoch_r3 = np.argmax(data_r3['val_acc'])
print(f"\nBest Val Accuracy - 15*15*15: {data_r1['val_acc'][best_epoch_r1]:.2f}% at epoch {best_epoch_r1+1}")
print(f"Best Val Accuracy - 26*26*26: {data_r3['val_acc'][best_epoch_r3]:.2f}% at epoch {best_epoch_r3+1}")

# 绘制学习率对比（可选，单独画）
fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(data_r1['epoch'], data_r1['lr'], label='15*15*15', color='blue', alpha=0.8)
ax.plot(data_r3['epoch'], data_r3['lr'], label='26*26*26', color='red', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.show()