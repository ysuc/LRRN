#net

import torch as t
import torchvision as tv
from torch import nn
import timm


class LRRN(nn.Module):
    def __init__(self) -> None:
        super(LRRN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=4,padding=1,stride=2)
        self.conv2 = nn.Conv2d(32,16,kernel_size=4,padding=1,stride=2)
        self.conv3 = nn.Conv2d(16,16,kernel_size=4,padding=1,stride=2)
        self.convT1 = nn.ConvTranspose2d(16,16,kernel_size=4,padding=1,stride=2)
        self.convT2 = nn.ConvTranspose2d(32,16,kernel_size=4,padding=1,stride=2)
        self.convT3 = nn.ConvTranspose2d(48,3,kernel_size=4,padding=1,stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(3)
        self.gn1 = nn.GroupNorm(num_channels=32, num_groups=32,eps=0.001)
        self.gn2 = nn.GroupNorm(num_channels=16, num_groups=16,eps=0.001)
        self.conv11 = nn.Sequential(
			nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
        self.conv12=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
        self.conv13=nn.Sequential(
			nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU()
		)
        self.conv14=nn.Sequential(
			nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5,stride=1,padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU()
		)
        self.out=nn.Sequential(nn.Dropout(0.6),
							   nn.Linear(64*64*32,45)
		)
        #self.pool=nn.MaxPool2d(kernel_size=2)


    def forward(self,x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn2(self.conv3(x2)))
        x5 = self.relu(self.bn2(self.convT1(x3)))
        x5 = t.cat((x2,x5),dim=1)
        x6 = self.relu(self.bn2(self.convT2(x5)))
        x6 = t.cat((x1,x6),dim=1)
        x7 = self.relu(self.convT3(x6))

        h=self.conv11(x7)
        #h=self.pool(h)
        h=self.conv13(h)
        h=self.conv12(h)
        #h=self.pool(h)
        h=self.conv14(h)
        h=h.view(-1,64*64*32)  #卷积块一维化(即每个数据都一维化，minibatch_size表示数据数量)
        o=self.out(h)
        return o