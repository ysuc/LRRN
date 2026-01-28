#modul_zoo

import torch as t
import torchvision as tv
from torch import nn


class CNH (nn.Module):   #设置用于完成分类head的类
    def __init__(self) -> None:
        super(CNH,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.out=nn.Sequential(nn.Dropout(0.6),
                               nn.Linear(16*16*32,45)
        )
        self.pool=nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool(c1)
        d1=self.conv3(p1)
        c2=self.conv2(d1)
        p2=self.pool(c2)
        d2=self.conv4(p2)
        h=d2.view(-1,16*16*32)  #卷积块一维化(即每个数据都一维化，minibatch_size表示数据数量)
        o=self.out(h)
        return o

class FIN(nn.Module):
    def __init__(self) -> None:
        super(FIN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=4,padding=1,stride=2)
        self.conv2 = nn.Conv2d(32,16,kernel_size=4,padding=1,stride=2)
        self.conv3 = nn.Conv2d(16,16,kernel_size=4,padding=1,stride=2)
        self.convT1 = nn.ConvTranspose2d(16,16,kernel_size=4,padding=1,stride=2)
        self.convT2 = nn.ConvTranspose2d(32,16,kernel_size=4,padding=1,stride=2)
        self.convT3 = nn.ConvTranspose2d(48,3,kernel_size=4,padding=1,stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self,x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn2(self.conv3(x2)))
        x5 = self.relu(self.bn2(self.convT1(x3)))
        x5 = t.cat((x2,x5),dim=1)
        x6 = self.relu(self.bn2(self.convT2(x5)))
        x6 = t.cat((x1,x6),dim=1)
        x7 = self.relu(self.convT3(x6))
        return x7
    

def resnet_18(num_classes):
    
    model = tv.models.resnet18(pretrained=False)  #决定是否使用预训练参数
    for parma in model.parameters():
        parma.requires_grad = True  #决定是否冻结预训练层,False为冻结预训练层
    model.fc = t.nn.Sequential(
        t.nn.Dropout(p=0.3),
        t.nn.Linear(512,num_classes)
    )
    return(model)

def resnet_50(num_classes):
    
    model = tv.models.resnet50(pretrained=False)  #决定是否使用预训练参数
    for parma in model.parameters():
        parma.requires_grad = True  #决定是否冻结预训练层,False为冻结预训练层
    model.fc = t.nn.Sequential(
        t.nn.Dropout(p=0.3),
        t.nn.Linear(2048,num_classes)
    )
    return(model)

def vgg_13(num_classes):
    
    model = tv.models.vgg13(pretrained=False)  #决定是否使用预训练参数
    for parma in model.parameters():
        parma.requires_grad = True  #决定是否冻结预训练层,False为冻结预训练层
    model.fc = t.nn.Sequential(
        t.nn.Dropout(p=0.3),
        t.nn.Linear(512,num_classes)
    )
    return(model)

class ICN_l(nn.Module):   #单通道ICN网络
    def __init__(self) -> None:
        super(ICN_l,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=4,padding=1,stride=2)
        self.conv2 = nn.Conv2d(32,16,kernel_size=4,padding=1,stride=2)
        self.conv3 = nn.Conv2d(16,16,kernel_size=4,padding=1,stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.convT1 = nn.ConvTranspose2d(16,16,kernel_size=4,padding=1,stride=2)
        self.convT2 = nn.ConvTranspose2d(32,16,kernel_size=4,padding=1,stride=2)
        self.convT3 = nn.ConvTranspose2d(32,3,kernel_size=4,padding=1,stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.pool(x3)
        x5 = self.relu(self.convT1(x4))
        x5 = t.cat((x3,x5),dim=1)
        x6 = self.relu(self.convT2(x5))
        x6 = t.cat((x2,x6),dim=1)
        x7 = self.relu(self.convT3(x6))
        return x7
    
class eCNN_l (nn.Module):   #设置用于完成分类head的类
    def __init__(self) -> None:
        super(eCNN_l,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),nn.MaxPool2d(kernel_size=2)
        )
        self.out=nn.Linear(1*1*32,2)
        self.pool=nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool(c1)
        c2=self.conv2(p1)
        p2=self.pool(c2)
        h=p2.view(-1,1*1*32)  #卷积块一维化(即每个数据都一维化，minibatch_size表示数据数量)
        o=self.out(h)
        return o
    

class ICN_nocat(nn.Module):
    def __init__(self) -> None:
        super(ICN_nocat,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=4,padding=1,stride=2)
        self.conv2 = nn.Conv2d(32,16,kernel_size=4,padding=1,stride=2)
        self.conv3 = nn.Conv2d(16,16,kernel_size=4,padding=1,stride=2)
        self.convT1 = nn.ConvTranspose2d(16,32,kernel_size=4,padding=1,stride=2)
        self.convT2 = nn.ConvTranspose2d(32,32,kernel_size=4,padding=1,stride=2)
        self.convT3 = nn.ConvTranspose2d(32,3,kernel_size=4,padding=1,stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self,x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn2(self.conv3(x2)))
        x5 = self.relu(self.bn1(self.convT1(x3)))
        x6 = self.relu(self.bn1(self.convT2(x5)))
        x7 = self.relu(self.convT3(x6))
        return x7
    
class FC(nn.Module):
    def __init__(self,ins,num_classes) -> None:
        super(FC,self).__init__()
        self.fc = t.nn.Sequential(
                  t.nn.Dropout(p=0.3),
                  t.nn.Linear(ins,num_classes))
    def forward(self,x):
        o = self.fc(x)
        return o