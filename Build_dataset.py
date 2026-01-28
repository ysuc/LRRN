#数据集划分程序

import os
import random
import math
import shutil #高级的文件，文件夹，压缩包的处理模块,也主要用于文件的拷贝

#  需要划分的数据集路径
old_data = '**Original data path**'
#  划分后的数据集的保存路径
data = '**Build a complete dataset save path**'
def data_split(old_path):
  new_path = data
  if os.path.exists(data) == 0: #判断括号里的文件是否存在的意思，括号内的可以是文件路径
    os.makedirs(new_path) #递归创建目录。
  for root_dir,sub_dirs,file in os.walk(old_path):
    for sub_dir in sub_dirs:  #遍历每个次级目录
      file_names = os.listdir(os.path.join(root_dir,sub_dir))
      #file_names = list(filter(lambda x: x.endswith('jpg'),file_names)) #去掉列表中的非jpg格式的文件
      random.shuffle(file_names) #将列表中的文件打乱

      for i in range(len(file_names)):
        if i < math.floor(0.8*len(file_names)): #将90%的图片设置为训练集
          sub_path = os.path.join(new_path,'train',sub_dir)

        elif i < len(file_names):        #将5%的图片设置为验证集
          sub_path = os.path.join(new_path,'val',sub_dir)
        if os.path.exists(sub_path) == 0:
          os.makedirs(sub_path)
        shutil.copy(os.path.join(root_dir,sub_dir,file_names[i]),os.path.join(sub_path,file_names[i]))
if __name__=='__main__':
  data_path = old_data
  data_split(data_path)