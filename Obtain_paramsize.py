#gat_paramsize
    
import modul_zoo as mz
import torch
from torchinfo import summary
import ViT

net = mz.resnet_50(45)

x = torch.rand(24,3,256,256)
out = net(x)
summary(net,(24,3,256,256))