#coding=gbk
import numpy as np
import torch
import os
import random
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader,Dataset
from wsnet_model import CNN2

import torchvision
import tarfile
from torchvision import transforms
import torch.nn as nn
from torch.nn import functional as F
from itertools import chain


class MyDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.data=os.listdir(self.root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):#__getitem__ 方法从数据集中获取索引为 index 的样本
        filename=self.data(index)
        wsnetpath=os.path.join(self.root_dir,filename)
        wsfeat=np.fromfile(wsnetpath,'float32')
        label=int(wsnetpath.split('/')[-1].split('_')[0]-1)

        # 数据归一化
        wsfeat=(wsfeat-np.min(wsfeat))/(np.max(wsfeat)-np.min(wsfeat))

        #转torch
        wsfeat=torch.from_numpy(wsfeat).float()
        label=[label]
        label=np.array(label)
        label=torch.from_numpy(label).long()

        return wsfeat,label
        #print('wsnetpath:',wsnetpath)


# 固定随机种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(27)


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("INFO: device = ", device)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 超参数设置
checkpoint = 'results/'
lr = 0.1
batchsize=50
steps = [100, 200]
train_size = 5000 * 0.7
test_size = 5000 * 0.3

#构建数据集
path = '../dataset/wsnet/g1/all'
pathtest = '../dataset/wsnet/g1/test'
dataset=MyDataset(path)

train_dataset1, test_dataset1 = torch.utils.data.random_split(dataset, [int(train_size), int(test_size)],
                                                                        generator=torch.Generator().manual_seed(0))
train_loader1 = DataLoader(train_dataset1, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)
test_loader1 = DataLoader(test_dataset1, batch_size=25, shuffle=False, num_workers=0, pin_memory=True)

#构建模型
cnn_model=CNN2(dataset)

#将模型放到GPU上
cnn_model.to(device)

