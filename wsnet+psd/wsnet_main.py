#coding=gbk
import numpy as np
import torch
import os
import random
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader,Dataset
from wsnet_model import CNN2
from wsnet_classifier import Classifier
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.cuda.device_count()
# os.environ['TORCH_USE_CUDA_DSA']

class MyDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.data=os.listdir(self.root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):#__getitem__ 方法从数据集中获取索引为 index 的样本
        # print('lendeata:',self.data[0])
        filename=self.data[index]
        wsnetpath=os.path.join(self.root_dir,filename)
        wsfeat=np.fromfile(wsnetpath,'float32',4000)
        label=int(wsnetpath.split('\\')[-1].split('_')[0])-1

        # 数据归一化
        wsfeat=(wsfeat-np.min(wsfeat))/(np.max(wsfeat)-np.min(wsfeat))
        wsfeat=wsfeat.reshape(1,len(wsfeat))

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

def main(args):
    # 设置随机数种子
    setup_seed(27)


    # 设备设置
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # print("INFO: device = ", device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # 超参数设置
    checkpoint = './results/model(cnn2)/'
    lr = 0.005
    gamma=0.1
    epoch=50
    batchsize=20
    steps = [100, 200]
    train_size = 4990 * 0.7
    test_size = 4990 * 0.3

    #构建数据集
    deviceNum=5
    cnn_in=1024
    path = '../dataset/wsnet/g1/all'
    pathtest = '..\\dataset\\wsnet\\g1\\test'
    dataset=MyDataset(path)

    train_dataset1, test_dataset1 = torch.utils.data.random_split(dataset, [int(train_size), int(test_size)],
                                                                            generator=torch.Generator().manual_seed(0))
    train_loader1 = DataLoader(train_dataset1, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)
    test_loader1 = DataLoader(test_dataset1, batch_size=25, shuffle=False, num_workers=0, pin_memory=True)
    # train_dataset1.to(device)
    # test_loader1.to(device)
    #构建模型
    cnn_model=CNN2(cnn_in,deviceNum)

    #将模型放到GPU上
    cnn_model.to(device)

    classification=Classifier(cnn_model,train_loader1,test_loader1,deviceNum)
    criterion=torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer=torch.optim.Adam(cnn_model.parameters(),args.lr,weight_decay=3e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,steps,gamma)
    classification.train(criterion,optimizer,args.epochs,scheduler)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100, help="Number of iteration")
    ap.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    args = ap.parse_args()
    print("========== arguments ==========")
    print(args)
    print("================================")
    main(args)