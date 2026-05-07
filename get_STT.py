import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import modul_zoo as mz 
import LRRN_modul as L
import ViT



def measure_single_iter_training_time(model, input_tensor, target, criterion, optimizer,
                                      device='cuda', num_warmup=10, num_runs=50):
    """
    测量模型在单张图像上执行一次完整训练迭代（前向+损失+反向+优化器 step）所需时间。

    Args:
        model (nn.Module): 待测模型（已置于训练模式）
        input_tensor (torch.Tensor): 输入图像，形状 (1, C, H, W)
        target (torch.Tensor): 对应的标签，形状 (1,) 或 (1, num_classes)
        criterion: 损失函数
        optimizer: 优化器
        device (str): 'cuda' 或 'cpu'
        num_warmup (int): 预热迭代次数（不计入统计）
        num_runs (int): 正式测量次数

    Returns:
        mean_time (float): 平均训练时间（毫秒）
        std_time (float): 标准差（毫秒）
        times (list): 每次测量的耗时列表（毫秒）
    """
    model = model.to(device).train()
    input_tensor = input_tensor.to(device)
    target = target.to(device)

    # 预热
    for _ in range(num_warmup):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 同步确保 GPU 操作完成
    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        # 注意：每次迭代开始前需要清零梯度（但这里测量耗时，已包含）
        optimizer.zero_grad()
        start = time.perf_counter()

        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # 转换为毫秒

    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time, times

# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    # 选择一个简单模型（如 ResNet18）
    import torchvision.models as models
    model = L.LRRN()   # 假设 45 类分类

    # 准备单张图像和标签（随机数据）
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    target = torch.randint(0, 45, (batch_size,))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.01)

    # 选择设备（优先 GPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测量单次训练迭代耗时
    mean_time, std_time, _ = measure_single_iter_training_time(
        model=model,
        input_tensor=input_tensor,
        target=target,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_warmup=10,
        num_runs=50
    )

    print(f"单次训练迭代平均耗时: {mean_time:.3f} ms")
    print(f"标准差: {std_time:.3f} ms")