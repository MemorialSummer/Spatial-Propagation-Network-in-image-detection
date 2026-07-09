import json
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
with open('visualization/training_history_15*15*15.json', 'r') as f:
    r1 = json.load(f)
with open('visualization/training_history_26*26*26.json', 'r') as f:
    r3 = json.load(f)

# 创建单张图
fig, ax = plt.subplots(figsize=(12, 7))

# 15*15*15: train细虚线, val粗实线
ax.plot(r1['epoch'], r1['train_acc'], label='15*15*15 (Train)', 
        color='blue', linewidth=1.0, linestyle='--', alpha=0.7)
ax.plot(r1['epoch'], r1['val_acc'], label='15*15*15 (Val)', 
        color='blue', linewidth=2.0)

# 26*26*26: train细虚线, val粗实线
ax.plot(r3['epoch'], r3['train_acc'], label='26*26*26 (Train)', 
        color='red', linewidth=1.0, linestyle='--', alpha=0.7)
ax.plot(r3['epoch'], r3['val_acc'], label='26*26*26 (Val)', 
        color='red', linewidth=2.0)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('LOCAL_RADIUS Comparison: Train vs Validation Accuracy', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

# 标注最佳点
best_r1_idx = np.argmax(r1['val_acc'])
best_r3_idx = np.argmax(r3['val_acc'])
ax.scatter(r1['epoch'][best_r1_idx], r1['val_acc'][best_r1_idx], 
           color='blue', s=80, zorder=5, marker='*')
ax.scatter(r3['epoch'][best_r3_idx], r3['val_acc'][best_r3_idx], 
           color='red', s=80, zorder=5, marker='*')
ax.annotate(f'Best: {r1["val_acc"][best_r1_idx]:.1f}%', 
            xy=(r1['epoch'][best_r1_idx], r1['val_acc'][best_r1_idx]),
            xytext=(10, 10), textcoords='offset points', color='blue', fontweight='bold')
ax.annotate(f'Best: {r3["val_acc"][best_r3_idx]:.1f}%', 
            xy=(r3['epoch'][best_r3_idx], r3['val_acc'][best_r3_idx]),
            xytext=(10, -20), textcoords='offset points', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印对比
print(f"Radius=1 最佳验证准确率: {r1['val_acc'][best_r1_idx]:.2f}% (Epoch {best_r1_idx+1})")
print(f"Radius=3 最佳验证准确率: {r3['val_acc'][best_r3_idx]:.2f}% (Epoch {best_r3_idx+1})")
print(f"差距: {r1['val_acc'][best_r1_idx] - r3['val_acc'][best_r3_idx]:.2f}%")