#coding=gbk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from model import CNN
import numpy as np

#参数设置
N=1024
class_n=5
ifCNN=True
batchSize=8
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集和模型
class MyDataSet():
    def __init__(self,row,class_num):
        dataset = []
        label = []
        for i in range(class_num):
            tmp = []
            if (row == 256):
                tmp = np.fromfile("../dataset/rmt_value/group_1_256/d", dtype=np.float32).reshape(-1)
            elif (row == 1024):
                tmp = np.fromfile("../dataset/rmt_value/group_a2_1024/d" + str(i + 1) + "_1024_1024.bin", dtype=np.float32)
            elif (row == 512):
                tmp = np.fromfile("../dataset/rmt_value/group_a1/d" + str(i + 1) + "_512_512.bin", dtype=np.float32)
            label = np.concatenate([label, (i * np.ones(int(len(tmp) / row))).reshape(-1)], axis=0)
            dataset = np.concatenate([dataset, tmp], axis=0)
        dataset = dataset.reshape(-1, 1, row)
        self.x=dataset
        self.y=np.round(label)
        self.row=row
    def __getitem__(self,index):
        return self.x[index,:,:].reshape(-1,1,self.row),self.y[index]
    def __len__(self):
        return len(self.y)

#模型设置
mySet=MyDataSet(N,class_n)
test_loader=DataLoader(mySet,batch_size=batchSize,shuffle=True,num_workers=1)
model1=CNN(N,class_n,ifConv=ifCNN)
model1.load_state_dict(torch.load('../results/CNN/test/cnn_100_1024.pth'))
model1.to(device)
model1.eval()#进入测试模式，不训练# 设置模型为评估模式

from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @p')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()



# 通过模型获取预测结果和真实标签
all_preds = []
all_labels = []

if __name__ == '__main__':

    test_acc=0

    for j, data in enumerate(test_loader):
        x=data[0].to(torch.float32)
        y=data[1].to(torch.long)
        x=x.view(-1,1,N)
        # x1=x.to(device)
        outputs = model1(x.to(device))#test_pred改为outputs
        _, preds = torch.max(outputs, 1)
        preds=preds.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

    # test_predictions = test_model.predict(test_features, batch_size=16)
    # print(test_predictions)
    plot_cm(y, preds)

