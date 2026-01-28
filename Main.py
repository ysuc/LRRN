#end_to_end-train_ICN+eCNN

import torch as t
import torchvision as tv
import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import modul_zoo as mz


class DefaultConfigs(object):

    data_dir = "**Dataset save path**"
    data_list = ["train","val"]

    lr = 0.1
    epochs = 50
    num_classes = 10
    image_size = 256
    batch_size = 16        
    channels = 3
    gpu = "1"
    use_gpu = t.cuda.is_available()

config = DefaultConfigs()

normalize = tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225]
                                    )

transform = {
    config.data_list[0]:tv.transforms.Compose(
        [tv.transforms.Resize([256,256]),tv.transforms.CenterCrop([256,256]),
        tv.transforms.ToTensor(),normalize]#tv.transforms.Resize 用于重设图片大小
    ) ,
    config.data_list[1]:tv.transforms.Compose(
        [tv.transforms.Resize([256,256]),tv.transforms.ToTensor(),normalize]
    ) 
}

datasets = {
    x:tv.datasets.ImageFolder(root = os.path.join(config.data_dir,x),transform=transform[x])
    for x in config.data_list
}

dataloader = {
    x:t.utils.data.DataLoader(dataset= datasets[x],
        batch_size=config.batch_size,
        shuffle=True
    ) 
    for x in config.data_list
}

def train(epochs):
    p_acc=[]
    p_loss=[]
    tracc = 0.0
    teacc = 0.0
    model1 = mz.FIN()
    model2 = mz.CNH()
    print(model1)
    print(model2)
    loss1 = t.nn.CrossEntropyLoss()
    if(config.use_gpu):
        model1 = model1.cuda()
        model2 = model2.cuda()
        loss1 = loss1.cuda()
    
    opt = t.optim.Adam([
	{'params': model1.parameters(), 'lr': config.lr,}, 
	{'params': model2.parameters()},
	])  #对两个模型的参数进行优化
    StepLR = t.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.90)  #设置学习率指数衰减
    time_start = time.time()
    
    for epoch in range(epochs):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        
        model1.train(True)
        model2.train(True)
        print("Epoch {}/{}".format(epoch+1,epochs))
        for batch, datas in tqdm(enumerate(iter(dataloader["train"]))):
            x,y = datas
            if (config.use_gpu):
                x,y = x.cuda(),y.cuda()
            y1 = model1(x)
            y2 = model2(y1)
            _, pre_y_ = t.max(y2,1)
            pre_y = y
            loss = loss1(y2,pre_y)  #计算分类损失
            acc = t.sum(pre_y_ == pre_y)
 
            loss.backward()
            opt.step()
            StepLR.step()
            opt.zero_grad()
            if(config.use_gpu):
                loss = loss.cpu()
                acc = acc.cpu()
            p_acc.append(tracc)
            p_loss.append(loss.data)
            train_loss.append(loss.data)
            train_acc.append(acc)
            #if((batch+1)%5 ==0):
            time_end = time.time()
            tracc = np.mean(train_acc)/config.batch_size
        print("Batch {}, Train loss:{:.4f}, Train acc:{:.4f}, Time: {}"\
            .format(batch+1,np.mean(train_loss)/config.batch_size,tracc,(time_end-time_start)))
        time_start = time.time()
        
        model1.train(False)  #验证部分
        model2.train(False)
        
        for batch, datas in tqdm(enumerate(iter(dataloader["val"]))):
            x,y = datas
            if (config.use_gpu):
                x,y = x.cuda(),y.cuda()
            y1 = model1(x)
            y2 = model2(y1)
            #print(x.shape,y.shape,y_.shape)
            _, pre_y_ = t.max(y2,1)

            pre_y = y
            #print(y_.shape)
            loss_t = loss1(y2,pre_y)
            acc = t.sum(pre_y_ == pre_y)
 
            if(config.use_gpu):
                loss_t = loss_t.cpu()
                acc = acc.cpu()
 
            test_loss.append(loss_t.data)
            test_acc.append(acc)
        time_end = time.time()
        teacc = np.mean(test_acc)/config.batch_size
        print("Batch {}, Test loss:{:.4f}, Test acc:{:.4f}, Time :{}".format(batch+1,np.mean(test_loss)/config.batch_size,teacc,(time_end-time_start)))
        time_start = time.time()
    #图像绘制
    '''plt.plot(p_acc, label='Acc')
    plt.plot(p_loss, label='Loss')

    plt.xlabel('step label')
    plt.ylabel('acc-loss label')

    plt.title("ICN+eCNN")
    plt.legend()
    plt.show()'''


    plt.subplot(211)  # the same as plt.subplot(2, 1, 1)
    plt.plot(p_loss,'g-')
    plt.ylabel('loss label')
    plt.title("Scene_recognition_task")

    plt.subplot(212)
    plt.plot(p_acc, 'y-')
    plt.xlabel('step label')
    plt.ylabel('acc label')

    plt.show()

    
    t.save(model1.state_dict(), '**FIN parameter save path**')   # 只保存网络中训练完成的参数
    t.save(model2.state_dict(), '**CNH parameter save path**')   # 只保存网络中训练完成的参数
 

 
if __name__ == "__main__":
    train(config.epochs)
