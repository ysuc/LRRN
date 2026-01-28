#get_confusion_matrix

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import modul_zoo as mz

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理（根据你的训练设置修改参数）
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集（修改为你的测试集路径）
test_dataset = ImageFolder(root='**Dataset save path**/val', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

# 加载模型（替换为你的模型定义）
model1 =mz.FIN().to(device)
model1.load_state_dict(torch.load('**FIN parameter save path**', map_location=torch.device('cpu')))
model1.eval()
model2 =mz.CNH().to(device)
model2.load_state_dict(torch.load('**FIN parameter save path**', map_location=torch.device('cpu')))
model2.eval()

# 收集预测结果
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs1 = model1(images)
        outputs = model2(outputs1)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 生成混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 行归一化

# 可视化设置
plt.rcParams.update({'font.size': 12})  # 统一字体大小
fig, ax = plt.subplots(figsize=(10, 8))
sns.set(font_scale=1.2)  # 调整seaborn字体缩放比例

# 绘制混淆矩阵
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            cbar=True, square=True, ax=ax,
            annot_kws={'size': 10},  # 调整注释字体大小
            linewidths=0.5, linecolor='gray')

# 修改后的混淆矩阵绘制部分
class_names = test_dataset.classes  # 获取类别名称
# 高级混淆矩阵可视化配置（45类别优化版）
plt.figure(figsize=(24, 20))  # 增大画布尺寸
ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",  # 改用高对比度配色
    cbar=True,
    square=False,    # 取消强制正方形以优化空间
    linewidths=0.1,  # 减小网格线宽度
    linecolor='#888888',
    annot_kws={
        'size': 12,   # 减小标注字体
        'color': '#333333',
        'va': 'center',
        'ha': 'center'
    },
    cbar_kws={
        'shrink': 0.75,  # 缩小颜色条
        'label': 'Sample Counts'
    }
)

# 坐标轴优化
ax.set_xticks(np.arange(len(class_names)) + 0.5)  # 精确对齐标签
ax.set_yticks(np.arange(len(class_names)) + 0.5)
ax.set_xticklabels(
    class_names,
    rotation=90,  # 垂直旋转标签
    ha='center',
    fontsize=12,
    fontfamily='Arial'
)
ax.set_yticklabels(
    class_names,
    rotation=0,  
    fontsize=12,
    fontfamily='Arial',
    va='center'
)

# 添加参考网格
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('#444444')
    spine.set_linewidth(0.5)

# 设置阈值参数（根据需求调整）
SAMPLE_COUNT_THRESHOLD = 50  # 当样本数超过该值时显示白色字体

# 获取扁平化后的混淆矩阵值
flat_cm = cm.flatten()

# 遍历所有注释文本并调整颜色
for k, text_obj in enumerate(ax.texts):
    # 确保不超过矩阵元素总数
    if k < flat_cm.size and flat_cm[k] > SAMPLE_COUNT_THRESHOLD:
        text_obj.set_color('white')
        text_obj.set_weight('bold')  # 可选：添加粗体强调

# 优化颜色条标签可见性
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color='white', labelcolor='black')  # 保持颜色条标签黑色

# 输出优化
plt.tight_layout(pad=2.0)  # 增加边距
plt.savefig(
    'Large_CM.pdf',
    dpi=600,
    bbox_inches='tight',
    metadata={
        'CreationDate': None,
        'Producer': None
    }
)

# 保存优化
plt.tight_layout(pad=3.0)  # 增加边距
plt.savefig('enhanced_confusion_matrix45.pdf', dpi=600, bbox_inches='tight')
plt.savefig('enhanced_confusion_matrix45.png', dpi=500, bbox_inches='tight')
plt.show()