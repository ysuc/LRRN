import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from modul_zoo import ICN3, eCNN  # 从你的模型文件导入

# ------------------ Grad-CAM 实现 ------------------
class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: 完整模型（包含 ICN3 和 eCNN）
            target_layer: 目标卷积层（例如 model.e_conv4[0]）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _compute_heatmap(self):
        # 权重：梯度在空间维度上的全局平均
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # 线性组合 + ReLU
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        # 归一化
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)  # 前向传播
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()    # 反向传播

        heatmap = self._compute_heatmap()
        # 上采样到输入图像尺寸
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # 将输入张量转为可显示图像（假设输入为 [0,1] 范围）
        img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # 生成彩色热力图并叠加
        heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_resized * 255), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        return heatmap_resized, superimposed

# ------------------ 构建完整模型 ------------------
class ICN3_eCNN(nn.Module):
    def __init__(self, icn3, ecnn):
        super().__init__()
        self.icn3 = icn3
        self.ecnn = ecnn

    def forward(self, x):
        # ICN3 的输出作为 eCNN 的输入
        x = self.icn3(x)
        x = self.ecnn(x)
        return x

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # 1. 实例化两个子网络
    icn3 = ICN3()
    ecnn = eCNN()

    # 2. 加载训练好的权重（请替换为实际文件路径）
    icn3.load_state_dict(torch.load("ICN3.pth", map_location='cpu'))
    ecnn.load_state_dict(torch.load("eCNN.pth", map_location='cpu'))

    # 3. 组合模型
    model = ICN3_eCNN(icn3, ecnn)
    model.eval()

    # 4. 定位 eCNN 的最后一个卷积层（conv4 内部第一个 Conv2d）
    target_layer = ecnn.conv4[0]   # Conv2d(64, 32, kernel_size=5, padding=2)
    print(f"目标层: {target_layer}")

    # 5. 输入图像预处理（eCNN 期望 64x64 输入，故整体输入也为 64x64）
    image_path = "your_image.jpg"   # 请替换为真实图片路径
    pil_img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),       # 缩放像素到 [0,1]
    ])
    input_tensor = preprocess(pil_img).unsqueeze(0)   # [1,3,64,64]

    # 6. 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 7. 生成 Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    heatmap, overlay = grad_cam(input_tensor)

    # 8. 显示结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(pil_img.resize((64, 64)))   # 显示缩放后的原图
    plt.title("Input (64x64)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 可选：保存叠加图
    # cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))