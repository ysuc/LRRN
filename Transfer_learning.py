#transfer_learning

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision as tv
import modul_zoo

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理（保持不变）
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(
    '**Transfer learning dataset save path**/train',
    transform=train_transforms
)
val_dataset = datasets.ImageFolder(
    '**Transfer learning dataset save path**/val',
    transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ------------------ 关键修改部分 -------------------
# 创建模型（输出类别设为5）
modelD1 = tv.models.convnext_base(pretrained=False)
pth_path1 = r'**Net parameter save path**'
#modelD2 = modul_zoo.CNH()
#pth_path2 = r'**CNH parameter save path**'

# 加载并过滤权重
state_dict1 = torch.load(pth_path1, map_location=device)
#state_dict2 = torch.load(pth_path2, map_location=device)

# 移除全连接层线性部分的权重（适配新类别数）
keys_to_remove = ['fc.1.weight', 'fc.1.bias']

# 加载过滤后的权重（严格模式关闭）
modelD1.load_state_dict(state_dict1, strict=False)
print(modelD1)
#modelD2.load_state_dict(state_dict2, strict=False)
#print(modelD2)


# ------------------------------------------------

# 将模型送入设备
model = modul_zoo.FC(ins=1000,num_classes=5).to(device)

# 定义优化器（仅优化全连接层）
optimizer = optim.Adam(model.parameters(), lr=0.001)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)  #设置学习率指数衰减

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练循环
num_epochs = 15
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        print(modelD1(images).shape)
        outputs = model(modelD1(images))
        loss = criterion(outputs, labels)
        loss.backward()  # 此处应能正常计算梯度
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    
    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(modelD1(images))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Val Acc: {val_acc:.4f}')
    
    if val_acc > best_acc:
        best_acc = val_acc
        #torch.save(model.state_dict(), 'LRRN_T_YSNR-5_best.pth')

print(f'Best Validation Accuracy: {best_acc:.4f}')