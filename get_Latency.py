import torch
import time
import numpy as np

def measure_latency(model, input_shape, device='cuda', num_warmup=50, num_runs=200, batch_size=1):
    """
    测量 PyTorch 模型的推理延迟（Latency）。

    Args:
        model (nn.Module): 待测模型，已置于评估模式。
        input_shape (tuple): 输入张量的形状（不包括 batch 维度），如 (3, 224, 224)。
        device (str): 'cuda' 或 'cpu'。
        num_warmup (int): 预热次数，使 GPU/CPU 进入稳定状态。
        num_runs (int): 正式测试次数。
        batch_size (int): 推理时的批量大小（影响延迟，通常与输入形状一致）。

    Returns:
        mean_latency (float): 平均延迟（毫秒）。
        std_latency (float): 标准差（毫秒）。
        all_latencies (list): 每次推理的延迟列表（毫秒）。
    """
    model = model.to(device).eval()
    
    # 创建随机输入，batch_size 作为第一个维度
    rand_input = torch.randn(batch_size, *input_shape).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(rand_input)
    
    # 如果使用 GPU，同步以确保计时准确
    if device == 'cuda':
        torch.cuda.synchronize()
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()  # 高精度计时
            _ = model(rand_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    return mean_latency, std_latency, latencies

# ---------- 使用示例 ----------
if __name__ == "__main__":
    import modul_zoo as mz
    import LRRN_modul as L
    import ViT
    
    # 加载模型（以 ResNet50 为例）
    model = L.LRRN()
    
    # 定义输入形状
    input_shape = (3, 256, 256)
    
    # 在 CPU 上测量
    device = 'cpu'
    mean_lat, std_lat, _ = measure_latency(model, input_shape, device=device, 
                                           num_warmup=30, num_runs=100)
    print(f"Device: {device.upper()}")
    print(f"Mean Latency: {mean_lat:.2f} ms")
    print(f"Std Latency:  {std_lat:.2f} ms")
    print("-" * 30)
    
    # 如果 CUDA 可用，在 GPU 上测量
    if torch.cuda.is_available():
        model_gpu = L.LRRN()
        mean_lat, std_lat, _ = measure_latency(model_gpu, input_shape, device='cuda',
                                               num_warmup=50, num_runs=200)
        print(f"Device: CUDA")
        print(f"Mean Latency: {mean_lat:.2f} ms")
        print(f"Std Latency:  {std_lat:.2f} ms")