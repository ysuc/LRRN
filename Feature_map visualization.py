
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import torch
import modul_zoo as mz

from PIL import Image
import os
import os.path

data_from=r'**Data source folder path**'   #指明数据来源文件夹

model = mz.FIN()
model.load_state_dict(torch.load('**FIN parameter save path**'))  #在模型中加载训练完成的参数

for parent, dirnames, filenames in os.walk(data_from):#遍历数据来源文件夹的每一张图片
    filenames.sort()
    for filename in filenames:
        
        currentPath = os.path.join(parent, filename)
        print('当前文件地址:' + currentPath)  #打印地址及文件名
   
        img = mpimg.imread(currentPath)  #读取图片
        img = np.transpose(img, [2, 0, 1])
        img = img[np.newaxis,:]  #增加一个空维度，使形状匹配
        img1 = torch.tensor(img)
        img_G = model(img1.to(torch.float32))
        #img_G = torch.tensor(img_G, dtype=torch.int16)
        img_G = img_G.detach().numpy()
        img_G = np.squeeze(img_G)
        img_G = np.transpose(img_G, [1, 2, 0])

        print(img_G) 
        plt.imshow(img_G)
        plt.show()  #显示图片
        