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

# modelPath = 'result/cnn_checkpoint_69.65%.pth' ##Ϊ�����洢��ѵ������ģ�Ͳ�����·��

class Data(Dataset):  # �̳�Dataset
    def __init__(self, root_dir):  # __init__�ǳ�ʼ�������һЩ��������
        self.root_dir = root_dir  # �ļ�Ŀ¼
        self.data = os.listdir(self.root_dir)  # Ŀ¼��������ļ�

    def __len__(self):  # �����������ݼ��Ĵ�С
        # print(len(self.data))
        return len(self.data)

    def __getitem__(self, index):  # ��������index����dataset[index]
        image_index = self.data[index]  # ��������index��ȡ��ͼƬ
        data_path = os.path.join(self.root_dir, image_index)  # ��ȡ����Ϊindex��ͼƬ��·����
        # data=np.fromfile(data_path,'float32')###########jjjj
        # data=np.loadtxt(data_path)
        with open(data_path) as f:##jjjj
            data = np.loadtxt(itertools.islice(f, 0, 2048), delimiter=',')  # ���Զ�ȡ���ǰ4500��
        label = int(data_path.split('\\')[-1].split('_')[0])-1  # ���ݸ�ͼƬ��·������ȡ��ͼƬ��label���������·�������зָ

        data = data.reshape(1, len(data))#(1,2048)

        # ��������PCAʹ��
        pca = PCA(n_components=0.95)  # ָ�������ɷ��ۼ���������ռ95%����Щ�ɷ�
        pca.fit(data.T)
        spectrum = pca.transform(data.T).T

        # ���ݹ�һ��
        # spectrum = (spectrum-np.mean(spectrum))/np.std(spectrum)
        spectrum = (spectrum-np.min(spectrum))/(np.max(spectrum)-np.min(spectrum))

        # תtorch
        spectrum = torch.from_numpy(spectrum).float()
        label = [label]
        label = np.array(label)
        label = torch.from_numpy(label).long()
        return spectrum, label

# ����������������������ݼ��õģ��̶�������ӿ��Ա�֤ÿ�δ��ҵ�˳����һ���ģ���֤�˽���ɸ���
# �̶��������
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    # �������������
    setup_seed(42)

    # random.seed(100);

    # Basic Info

    ## ����Ƿ��ܹ�ʹ��GPU
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda")            ####�ǲ���Ҫ�ĳ�cuda������������������������������

    print("INFO: device = ", device)
    # torch.manual_seed(0)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    checkpoint = 'result/'
    checkpoint='./results/model(cnn2)/'
    gamma = 0.1
    steps = [100, 200]
    train_size = 4990 * 0.8    #ѵ�����Ĵ�С
    test_size = 4990 * 0.2     #���Լ��Ĵ�С

    #dataset_path = 'D:\gkc_project\data\psd'
    dataset_path = r'E:\RFF\me\WSNet\data\S\S_800_2048'
    # dataset_path=r'D:\1.RFF\С����\python\dataset\wsnet\g1\all'
    # dataset_path='../dataset/wsnet/g1/all'

    custom_dataset = Data(dataset_path)

    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [int(train_size), int(test_size)])
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=0, pin_memory=True)

#######################################����model###################################################
    model = CNN(classes=5)  #��model.py����CNN��Ķ��壬classes=5��ζ�ŷֳ�5��
    # model = ResNet(Bottleneck, [2, 3, 5, 2], num_classes=5, include_top=True)
    #
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    model.to(device)

    classification = Classifier(model, train_loader, test_loader, device)

    # logSoftMax����NLL��SoftMax����Cross
    criterion = torch.nn.CrossEntropyLoss()  # acc=70%
    # acriterion = torch.nn.NLLLoss()  # acc=66%
    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=gamma)

    classification.train(criterion, optimizer, args.epochs, scheduler)




# Argparse��ָ������
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