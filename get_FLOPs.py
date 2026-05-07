import torch
import torch.nn as nn
import LRRN_modul as L
import modul_zoo as M
import ViT

import thop

def compute_flops(model, input_shape, device='cuda', as_macs=False):
    """
    计算 PyTorch 模型的 FLOPs 和参数量。

    Args:
        model (nn.Module): 待评估的模型。
        input_shape (tuple): 输入张量的形状（不包括 batch 维度），例如 (3, 224, 224)。
        device (str): 'cuda' 或 'cpu'。
        as_macs (bool): 若为 True，返回 MACs（乘加次数）；若为 False，返回 FLOPs（通常 = 2 * MACs）。

    Returns:
        flops (float): 浮点运算次数（单位：GFLOPs 或 MFLOPs 等，默认返回实际数值）。
        params (float): 参数量（单位：百万，默认返回实际数值）。
    """
    model = model.to(device).eval()
    # 创建随机输入，batch_size 通常设为 1
    x = torch.randn(1, *input_shape).to(device)

    # thop.profile 返回 MACs 和参数量
    macs, params = thop.profile(model, inputs=(x,), verbose=False)

    if as_macs:
        flops = macs
    else:
        flops = macs * 2   # 通常一个 MAC 含一次乘法和一次加法，算作 2 FLOPs

    return flops, params

# ---------- 使用示例 ----------
if __name__ == "__main__":
    import torchvision.models as models
    model = L.LRRN() # 替换为你想评估的模型
    flops, params = compute_flops(model, (3, 256, 256), device='cpu', as_macs=False)
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")