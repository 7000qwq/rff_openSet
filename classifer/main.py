#coding=gbk
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.nn as nn
import torch
import numpy as np
from model import CNN
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
# from sklearn.metrics import confusion_matrix

os.chdir(r"D:\Users\7000qwq\Downloads\rff\python\classifer")

NUM=256
N = NUM
M = NUM
class_train = 4 #Devices number
class_test = 5

c=N/M
batchSize = 8
numEpoch=101
check_step=10
learning_rate_max=1e-2
learning_rate_min=1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ifCNN=True#False
upper_limit=4

# 结果保存目录

serial=""
if ifCNN:
    # serial="../results/CNN/me/g1__multipath/3path_f2/cnn_"
    # serial = "../results/CNN/me/g1/-9dB/cnn_"
    # serial = "../results/CNN/wsnet/S_800/cnn_"
    serial='../results/psd/cnn_'

else:
    serial="./MLP/mlp_"

class MyDataSet():
    def __init__(self,row,class_num):
        dataset = []
        label = []
        for i in range(class_num):
            tmp = []
            if (row == 256):#"./day1/d"#原始256_256
                # 每次训练只用5个bin文件 训练10个不同db下的模型
                tmp = np.fromfile("../dataset/rmt_value/group_1000_256/d" + str(i+1) + "_256_256.bin", dtype=np.float32).reshape(-1)
            elif (row == 1024):
                tmp = np.fromfile("../dataset/rmt_value/group_a2_1024/d" + str(i+1) + "_1024_1024.bin", dtype=np.float32)
            elif (row == 512):
                # tmp = np.fromfile('../dataset/org_signal/me/g1_/512mod/d' + str(i + 1) + "_.bin", dtype=np.float32)
                # tmp = np.fromfile('../dataset/psd/g1/psd(new)_512mod_d' + str(i + 1) + "_.bin", dtype=np.float32)
                tmp= np.loadtxt('../dataset/psd/g1/psd(new)_512mod_d' + str(i + 1) + '.txt')
                # tmp = np.fromfile('../dataset/rmt_value/SNR/-9/-9dB_d' + str(i+1) + "_512_512.bin", dtype=np.float32)
                # tmp = np.fromfile('../dataset/rmt_value/me_multipath/3path_f2/3path_d' + str(i + 1) + "_512_512.bin", dtype=np.float32)
                # tmp = np.fromfile('../dataset/wsnet/S_800_bin/' + str(i + 1) + "_800_512mod.bin",dtype=np.float32)
            # elif (row == 4090):
                # tmp=np.fromfile()
            label = np.concatenate([label, (i * np.ones(int(len(tmp) / row))).reshape(-1)], axis=0)
            dataset = np.concatenate([dataset, tmp], axis=0)
        dataset = dataset.reshape(-1, 1, row)
        self.x=dataset
        self.y=np.round(label)#round是取整
        self.row=row
    def __getitem__(self,index):
        return self.x[index,:,:].reshape(-1,1,self.row),self.y[index]
    def __len__(self):
        return len(self.y)


class MyTestSet():
    def __init__(self,row,class_num):
        dataset = []
        label = []
        for i in range(class_num - 4):
            tmp = []
            if (row == 256):#"./day1/d"#原始256_256
                # 每次训练只用5个bin文件 训练10个不同db下的模型
                tmp = np.fromfile("../dataset/rmt_value/group_1000_256/d" + str(i+5) + "_256_256.bin", dtype=np.float32).reshape(-1)
            elif (row == 1024):
                tmp = np.fromfile("../dataset/rmt_value/group_a2_1024/d" + str(i+5) + "_1024_1024.bin", dtype=np.float32)

            label = np.concatenate([label, ((i + 4) * np.ones(int(len(tmp) / row))).reshape(-1)], axis=0)
            dataset = np.concatenate([dataset, tmp], axis=0)
        dataset = dataset.reshape(-1, 1, row)
        self.x=dataset
        self.y=np.round(label)#round是取整
        self.row=row
    def __getitem__(self,index):
        return self.x[index,:,:].reshape(-1,1,self.row),self.y[index]
    def __len__(self):
        return len(self.y)
    
torch.manual_seed(1926)
mySet=MyDataSet(N,class_train)
train_size = int(0.7 * len(mySet))
validation_size = len(mySet) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(mySet, [train_size, validation_size])
train_loader=DataLoader(train_dataset,batch_size=batchSize,shuffle=True,num_workers=1)
validation_loader=DataLoader(validation_dataset,batch_size=batchSize,shuffle=True,num_workers=1)

# test set
myTestSet=MyTestSet(N,class_test)
train_size = int(0.55 * len(myTestSet))
validation_size = len(myTestSet) - train_size
useless_dataset, unknow_dataset = torch.utils.data.random_split(myTestSet, [train_size, validation_size])
unknow_loader = DataLoader(unknow_dataset, batch_size=batchSize, shuffle=False, num_workers=1)

#模型
model=CNN(N,class_train,ifConv=ifCNN)
model.to(device)#移到gpu上训练

#损失函数
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)

#优化器
optimizer=optim.Adam(model.parameters(), lr=learning_rate_max)
accuracy_train=[]
accuracy_validation=[]
accuracy_unknow = []
loss=[]

# 如果温度大于 1，输出的概率分布将变得更平滑；如果温度小于 1，输出的概率分布将变得更尖锐
def softmax(X, temperature=1.0):
    # 通过温度参数调整输入
    X_temp = X / temperature
    X_exp = torch.exp(X_temp)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

if __name__ == '__main__':
    torch.manual_seed(42)
    for i in range(numEpoch):
        #初始化
        train_acc = 0
        train_loss = 0
        validation_acc=0
        unknow_acc = 0
        #训练
        model.train()

        for j, data in enumerate(train_loader):
            x=data[0].to(torch.float32)#数据类型转换
            y=data[1].to(torch.long)
            # print('xshape:', x.shape)
            # print('yshape:', y.shape)
            x=x.view(-1,1,N)#view():用于改变张量的形状
            optimizer.zero_grad()
            train_pred = model(x.to(device))
            # print('xpredshape:', train_pred.shape)

            # print(train_pred.shape) #torch.Size([8, 40, 448, 448])
            # print(y.to(device).shape) #torch.Size([8, 448, 448])
            batch_loss = criterion(train_pred, y.to(device))
            batch_loss.backward()
            # for param in model.parameters():
                # print("!!!!")
                # print(param.grad)  # 打印参数的梯度

            optimizer.step()
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == y.numpy())
            train_loss += batch_loss.item()

        model.eval()

        # y_gt = []
        # y_pred = []
        for j, data in enumerate(validation_loader):
            x=data[0].to(torch.float32)
            y=data[1].to(torch.long)
            x=x.view(-1,1,N)
            validation_pred = model(x.to(device))
            validation_acc += np.sum(np.argmax(validation_pred.cpu().data.numpy(), axis=1) == y.numpy())
            print('---------------pred-------------------\n',np.argmax(validation_pred.cpu().data.numpy(), axis=1))
            print('--------------actual------------------\n',y.numpy())

            # _, predict_np = torch.max(validation_pred, 1)
            # predict_np = predict_np.cpu().numpy()
            # y_pred.extend(predict_np.tolist())
            # y_gt.extend(y.tolist())

        for j, data in enumerate(unknow_loader):
            x=data[0].to(torch.float32)
            y=data[1].to(torch.long)
            x=x.view(-1,1,N)
            unknow_pred = softmax(model(x.to(device)))
            print(unknow_pred)
            max_values, _ = torch.max(unknow_pred, dim=1)
            # 设置阈值
            threshold = 0.9
            # 统计小于阈值的元素个数
            count = torch.sum(max_values < threshold).item()
            unknow_acc += count
            print('count:',count)

        print("Epoch ",i,"train accuracy ",train_acc/train_dataset.__len__()," validation accuracy ",validation_acc/validation_dataset.__len__(),"LR",optimizer.param_groups[0]['lr'], "loss", train_loss)
        optimizer.param_groups[0]['lr']=max(optimizer.param_groups[0]['lr']*0.95,learning_rate_min)
        accuracy_train.append(train_acc / train_dataset.__len__())
        accuracy_validation.append(validation_acc / validation_dataset.__len__())
        accuracy_unknow.append(unknow_acc / unknow_dataset.__len__())
        loss.append(train_loss)

        #保存
        if (i%check_step==0 and i/check_step>1):
            torch.save(model.state_dict(),serial+str(i)+"_"+str(N)+".pth")

    accuracy_train=np.array(accuracy_train)
    np.savetxt('../results/acc_loss/psd/train_acc_me_g1_512psd.csv', accuracy_train, fmt='%f', delimiter=None)  # frame: 文件 array:存入文件的数组 # fmt:写入文件的格式，如%d   %f   %e delimiter:分割字符串，默认空格
    accuracy_validation=np.array(accuracy_validation)
    np.savetxt('../results/acc_loss/psd/validation_acc_me_g1_512psd.csv', accuracy_validation, fmt='%f', delimiter=None)
    loss=np.array(loss)

    plt.figure(1)
    plt.plot(accuracy_train,label="Train")
    plt.plot(accuracy_validation,label="Validation")
    plt.plot(accuracy_unknow,label="Unknow")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title('Accuracy curve')

    plt.figure(2)
    plt.plot(loss)
    plt.xlabel("Epochs")
    plt.ylabel("Total CrossEntropy Loss per Epoch")
    plt.title('Loss curve')
    plt.show()











