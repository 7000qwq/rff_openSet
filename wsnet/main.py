# coding=gbk
import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import random
import argparse
from sklearn.decomposition import PCA

from classifier import *
from model import *
from matplotlib import pyplot as plt
# from resnet import *

SNR=6
path='../dataset/wsnet/SNR/g2/'+str(SNR)+'/'
# path='../dataset/wsnet/g1/SNR_'+str(SNR)+'/'
# path='../dataset/wsnet/SNR/g2/norm/'
filename=os.listdir(path)
for j in range(len(filename)):
    filepath=path+filename[j]
    data=np.fromfile(filepath,'float32')
    if len(data)<4090 or len(data)>4090:
        filenum=filename[j]
        print(f'{filenum},len{len(data)}')

test='snr_'+str(SNR)
print(test)
modelPath = 'results/cnn_checkpoint_69.65%.pth'  ##为期望存储的训练过的模型参数的路径


class Data(Dataset):  # 继承Dataset
    def __init__(self, root_dir):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.spectrum = os.listdir(self.root_dir)  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.spectrum)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.spectrum[index]  # 根据索引index获取该图片
        spectrum_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名

        spectrum = np.fromfile(spectrum_path, 'float32')
        # spectrum = spectrum * 1000 / 1024;
        # print(spectrum.shape)
        # label = int(spectrum_path.split('\\')[-1].split('_')[0]) # 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。
        label = int(spectrum_path.split('\\')[-1].split('_')[0]) - 1  # 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。
        # label = int(spectrum_path.split('/')[-1].split('\\')[-1].split('_')[1].split('.')[0])-1# 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。
        # spectrum = normalize(spectrum, 0.1440, 0.1440)  # 对样本进行变换

        spectrum = spectrum.reshape(1, 4090)
        # 数据归一化
        spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

        # 非正常的PCA使用
        pca = PCA(n_components=0.95)  # 指定了主成分累加起来至少占95%的那些成分
        pca.fit(spectrum.T)
        spectrum = pca.transform(spectrum.T).T

        # 数据归一化
        # spectrum = (spectrum-np.mean(spectrum))/np.std(spectrum)

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
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda")  ####是不是要改成cuda？？？？？？？？？？？？？？？

    print("INFO: device = ", device)
    # torch.manual_seed(0)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    checkpoint = 'result_test/'
    gamma = 0.1
    steps = [100, 200]
    train_size = 5000 * 0.7  # 训练集的大小
    test_size = 5000 * 0.3  # 测试集的大小


    #g2
    dataset_path='../dataset/wsnet/SNR/g2/'+str(SNR)
    modelsave='./results/model/SNR/'+str(SNR)

    # #g1
    # dataset_path='../dataset/wsnet/g1/SNR_'+str(SNR)
    # modelsave='./results/model/g1/SNR_'+str(SNR)
    # dataset_path='../dataset/wsnet/SNR/g2/norm'
    # modelsave='./results/model/SNR/norm'

    custom_dataset = Data(dataset_path)

    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [int(train_size), int(test_size)])
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=0, pin_memory=True)

    #######################################定义model###################################################
    model = CNN(classes=5)  # 在model.py中有CNN类的定义，classes=5意味着分成5类
    # model = ResNet(Bottleneck, [2, 3, 5, 2], num_classes=5, include_top=True)

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

    losses,test_accuracies,train_accuracies=classification.train(criterion, optimizer, args.epochs, scheduler, modelpath=modelsave)
    plt.figure(1)
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Total CrossEntropy Loss per Epoch')
    plt.title('Loss curve')
    plt.savefig('./results/pic/SNR/snr'+str(SNR)+'_loss.png')
    # plt.savefig('./results/pic/SNR/norm_loss.png')
    # plt.savefig('./results/pic/g1/snr' + str(SNR) + '_loss.png')

    plt.figure(2)
    labeltrain='snr'+str(SNR)+'_train_acc'
    labeltest='snr'+str(SNR)+'_test_acc'
    # labeltrain='train'
    # labeltest='test'
    plt.plot(train_accuracies,label=labeltrain)
    plt.plot(test_accuracies,label=labeltest)
    plt.title('Accuracy curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./results/pic/SNR/snr'+str(SNR)+'_acc.png')
    # plt.savefig('./results/pic/SNR/norm_acc.png')
    # plt.savefig('./results/pic/g1/snr' + str(SNR) + '_acc.png')


# Argparse可指定参数
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100, help="Number of iteration")
    ap.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = ap.parse_args()
    print("========== arguments ==========")
    print(args)
    print("================================")
    main(args)