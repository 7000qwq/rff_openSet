#coding=gbk
import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import random
import argparse
import itertools
from sklearn.decomposition import PCA

from aclassifier import Classifier
from amodel import CNN
from resnet import *

# modelPath = 'result/cnn_checkpoint_69.65%.pth' ##为期望存储的训练过的模型参数的路径

class Data(Dataset):  # 继承Dataset
    def __init__(self, root_dir):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.data = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        # print(len(self.data))
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.data[index]  # 根据索引index获取该图片
        data_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        # data=np.fromfile(data_path,'float32')###########jjjj
        # data=np.loadtxt(data_path)
        with open(data_path) as f:##jjjj
            data = np.loadtxt(itertools.islice(f, 0, 2048), delimiter=',')  # 可以读取大概前4500行
        label = int(data_path.split('\\')[-1].split('_')[0])-1  # 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。

        data = data.reshape(1, len(data))#(1,2048)

        # 非正常的PCA使用
        pca = PCA(n_components=0.95)  # 指定了主成分累加起来至少占95%的那些成分
        pca.fit(data.T)
        spectrum = pca.transform(data.T).T

        # 数据归一化
        # spectrum = (spectrum-np.mean(spectrum))/np.std(spectrum)
        spectrum = (spectrum-np.min(spectrum))/(np.max(spectrum)-np.min(spectrum))

        # 转torch
        spectrum = torch.from_numpy(spectrum).float()
        label = [label]
        label = np.array(label)
        label = torch.from_numpy(label).long()
        return spectrum, label

# 随机数种子是用来打乱数据集用的，固定随机种子可以保证每次打乱的顺序是一样的，保证了结果可复现
# 固定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    # 设置随机数种子
    setup_seed(42)

    # random.seed(100);

    # Basic Info

    ## 检测是否能够使用GPU
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda")            ####是不是要改成cuda？？？？？？？？？？？？？？？

    print("INFO: device = ", device)
    # torch.manual_seed(0)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    checkpoint = 'result/'
    checkpoint='./results/model(cnn2)/'
    gamma = 0.1
    steps = [100, 200]
    train_size = 4990 * 0.8    #训练集的大小
    test_size = 4990 * 0.2     #测试集的大小

    #dataset_path = 'D:\gkc_project\data\psd'
    dataset_path = r'E:\RFF\me\WSNet\data\S\S_800_2048'
    # dataset_path=r'D:\1.RFF\小论文\python\dataset\wsnet\g1\all'
    # dataset_path='../dataset/wsnet/g1/all'

    custom_dataset = Data(dataset_path)

    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [int(train_size), int(test_size)])
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=0, pin_memory=True)

#######################################定义model###################################################
    model = CNN(classes=5)  #在model.py中有CNN类的定义，classes=5意味着分成5类
    # model = ResNet(Bottleneck, [2, 3, 5, 2], num_classes=5, include_top=True)
    #
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    model.to(device)

    classification = Classifier(model, train_loader, test_loader, device)

    # logSoftMax搭配NLL，SoftMax搭配Cross
    criterion = torch.nn.CrossEntropyLoss()  # acc=70%
    # acriterion = torch.nn.NLLLoss()  # acc=66%
    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=gamma)

    classification.train(criterion, optimizer, args.epochs, scheduler)




# Argparse可指定参数
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100, help="Number of iteration")
    ap.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    args = ap.parse_args()
    print("========== arguments ==========")
    print(args)
    print("================================")
    main(args)

    train_acc = 0
    train_loss = 0
    test_acc = 0
    print('y2:')

    # accuracy_train = np.array(accuracy_train)
    # accuracy_test = np.array(accuracy_test)
    # loss = np.array(loss)
    # plt.figure(1)
    # plt.plot(accuracy_train, label="Train")
    # plt.plot(accuracy_test, label="Test")
    # plt.legend()
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.figure(2)
    # plt.plot(loss)
    # plt.xlabel("Epochs")
    # plt.ylabel("Total CrossEntropy Loss per Epoch")
    # plt.show()