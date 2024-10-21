#coding=gbk
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.nn as nn
import torch
import numpy as np
from wsnet_model import CNN2
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix



class MyDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.data=os.listdir(self.root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):#__getitem__ ���������ݼ��л�ȡ����Ϊ index ������
        # print('lendeata:',self.data[0])
        filename=self.data[index]
        wsnetpath=os.path.join(self.root_dir,filename)
        wsfeat=np.fromfile(wsnetpath,'float32')
        label=int(wsnetpath.split('\\')[-1].split('_')[0])-1

        # ���ݹ�һ��
        wsfeat=(wsfeat-np.min(wsfeat))/(np.max(wsfeat)-np.min(wsfeat))
        wsfeat=wsfeat.reshape(1,len(wsfeat))

        #תtorch
        wsfeat=torch.from_numpy(wsfeat).float()
        label=[label]
        label=np.array(label)
        label=torch.from_numpy(label).long()

        return wsfeat,label
        #print('wsnetpath:',wsnetpath)

#�����豸�Ȼ�������
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize=50
epoch=100
lr=0.005
check_step=10
innum=512*3
outnum=5#��������
torch.manual_seed(1926)# ����������������������ݼ��õģ��̶�������ӿ��Ա�֤ÿ�δ��ҵ�˳����һ���ģ���֤�˽���ɸ���
savepath='./results/cnn/cnn_'

#�������ݼ�
datapath='../dataset/wsnet/g1/all'
dataset=MyDataset(datapath)
trainSize=0.7*4990
testSize=0.3*4990
train_dataset,test_dataset=torch.utils.data.random_split(dataset,[int(trainSize), int(testSize)])
train_loader=DataLoader(train_dataset,batch_size=batchsize,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batchsize,shuffle=True)

#����ģ��
modelcnn=CNN2(innum,outnum)
modelcnn.to(device)

#��ʧ����
criterion=torch.nn.CrossEntropyLoss()
criterion.to(device)
#�Ż���
optimizer=optim.Adam(modelcnn.parameters(),lr)
#׼ȷ�ʺ���ʧ����
acc_train=[]
acc_test=[]
loss=[]

if __name__=='__main__':
    torch.manual_seed(27)
    for i in range(epoch):
        modelcnn.train()
        train_acc=0
        train_loss=0
        test_acc=0

        for j, data in enumerate(train_loader):
            x = data[0].to(torch.float32)  # ��������ת��
            y = data[1].to(torch.long)
            y=y.squeeze(1)
            # print('xshape:',x.shape)
            # print('yshape:',y.shape)
            x = x.view(-1, 1, 4090)  # view():���ڸı���������״
            optimizer.zero_grad()
            train_pred = modelcnn(x.to(device))
            # print('xpredshape:',train_pred.shape)

            # print(train_pred.shape) #torch.Size([8, 40, 448, 448])
            # print(y.to(device).shape) #torch.Size([8, 448, 448])
            batch_loss = criterion(train_pred, y.to(device))
            batch_loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == y.numpy())
            train_loss += batch_loss.item()

        modelcnn.eval()

        # y_gt = []
        # y_pred = []
        for j, data in enumerate(test_loader):
            x = data[0].to(torch.float32)
            y = data[1].to(torch.long)
            x = x.view(-1, 1, 4090)
            test_pred = modelcnn(x.to(device))
            test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == y.numpy())
            # print('---------------pred-------------------\n', np.argmax(test_pred.cpu().data.numpy(), axis=1))
            # print('--------------actual------------------\n', y.numpy())

            # _, predict_np = torch.max(test_pred, 1)
            # predict_np = predict_np.cpu().numpy()
            # y_pred.extend(predict_np.tolist())
            # y_gt.extend(y.tolist())

        print("Epoch ", i, "train accuracy ", train_acc / train_dataset.__len__(), " test accuracy ",
              test_acc / test_dataset.__len__(), "LR", optimizer.param_groups[0]['lr'])
        optimizer.param_groups[0]['lr'] = max(optimizer.param_groups[0]['lr'] * 0.95, lr)
        acc_train.append(train_acc / train_dataset.__len__())
        acc_test.append(test_acc / test_dataset.__len__())
        loss.append(train_loss)

        # ����
        if (i % check_step == 0 and i / check_step > 1):
            torch.save(modelcnn.state_dict(), savepath + str(i) + "_" + str(innum) + ".pth")

    accuracy_train = np.array(acc_train)
    np.savetxt('./results/acc_loss/loss.csv', accuracy_train, fmt='%f',
               delimiter=None)  # frame: �ļ� array:�����ļ������� # fmt:д���ļ��ĸ�ʽ����%d   %f   %e delimiter:�ָ��ַ�����Ĭ�Ͽո�
    accuracy_test = np.array(acc_test)
    np.savetxt('./results/acc_loss/test.csv', accuracy_test, fmt='%f', delimiter=None)
    loss = np.array(loss)

    plt.figure(1)
    plt.plot(accuracy_train, label="Train")
    plt.plot(accuracy_test, label="Test")
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
