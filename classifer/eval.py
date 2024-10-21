#coding=gbk
import torch.nn as nn
import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import CNN

N = 1024
M= 1024
class_n=5 #device number need to be matched with the actual data
beta=2
c=N/M
batchSize = 8
numEpoch=121
check_step=10
learning_rate_max=1e-4
learning_rate_min=1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ifCNN=True
upper_limit=4
serial=""
if ifCNN:
    serial="../results/CNN/group_a1_512/cnn_"
else:
    serial="./MLP_model/mlp_"

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


mySet=MyDataSet(N,class_n)
test_loader=DataLoader(mySet,batch_size=batchSize,shuffle=True,num_workers=1)
model1=CNN(N,class_n,ifConv=ifCNN)
model1.load_state_dict(torch.load('../results/CNN/test/cnn_100_1024.pth'))
# model(cnn2).load_state_dict(torch.load('./wired_model/cnn_100_512.pth'))
model1.to(device)
model1.eval()#进入测试模式，不训练

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None,
                          dpi=100):
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)
    plt.tight_layout()
    plt.colorbar()
    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)
    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @p')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':
    test_acc=0
    # model.eval()
    for j, data in enumerate(test_loader):
        x=data[0].to(torch.float32)
        y=data[1].to(torch.long)
        x=x.view(-1,1,N)
        # x1=x.to(device)
        test_pred = model1(x.to(device))
        test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == y.numpy())

        # confusion maxtric
        y_gt = []
        y_pred = []
        print(' batch:{0}\n x_data:{1}\nlabel: {2}'.format(j, x, y))  # y是labels
        predict_np = np.argmax(test_pred.cpu().detach().numpy(), axis=-1)  # array([0,5,1,6,3,...],dtype=int64)
        labels_np = y.numpy()  # array([0,5,0,6,2,...],dtype=int64)
        y_pred.append(predict_np)
        y_gt.append(labels_np)

    # draw_confusion_matrix(label_true=y_gt,  # y_gt=[0,5,1,6,3,...]
    #                       label_pred=y_pred,  # y_pred=[0,5,1,6,3,...]
    #                       label_name=["d1", "d2", "d3", "d4", "d5"],
    #                       title="Confusion Matrix",
    #                       pdf_save_path="Confusion_Matrix",
    #                       dpi=300)
    #
    # test_predictions = test_model.predict(test_features, batch_size=16)
    # # print(test_predictions)
    #
    # plot_cm(test_labels, test_predictions)

    test_acc/=len(mySet)
    print("Accuracy: ",test_acc)









