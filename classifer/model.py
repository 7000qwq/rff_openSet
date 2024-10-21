#coding=gbk
import torch.nn as nn
import torch
# from torchviz import make_dot
import numpy as np

N=512 #256#128#256
M=512 #256#128#1024
c=N/M


class CNN(torch.nn.Module):
    def __init__(self, in_num, out_num,ifConv=False):
        super(CNN, self).__init__()
        self.ifConv=ifConv
        self.out_num=out_num
        self.in_num=in_num
        st = 4#st是步长
        if(in_num==1024):
            st=4
        if(in_num==256):
            st=2
        elif(in_num==512):
            st=4
        self.conv=nn.Sequential(
            nn.Conv1d(1,8,kernel_size=4,stride=2,padding=2),
            # nn.BatchNorm1d(8),#
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=8,stride=4),#k=4,s=2

            nn.Conv1d(8, 64 , kernel_size=4, stride=2, padding=2),
            # nn.BatchNorm1d(64),#
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),

            nn.Conv1d(64, 256, kernel_size=4, stride=2, padding=2),
            # nn.BatchNorm1d(256),#256org
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4, stride=st),#这里stride原来=st

            # nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=2),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),#这里计算出的最后的n要满足：batchsize*n*convlast(out_num)=k*batchsize的整数倍
        )

        self.MLP =nn.Sequential(
            nn.Linear(in_num, 128),#(in_num, 512)
            nn.LeakyReLU(),
            nn.Linear(128, 128),#(512, 512)
            nn.LeakyReLU(),
            nn.Linear(128, 64),#(512, 128)
            nn.LeakyReLU(),
            nn.Linear(64, out_num)#(128, out_num)
        )
    def forward(self, x):
        if self.ifConv:
            x=self.conv(x)
            # print("Shape after conv:", x.shape)
            x=x.view(-1,self.in_num)
            # print("Shape after conv.view:", x.shape)
        x = self.MLP(x)
        # print('shape after mlp:',x.shape)
        x=x.view(-1,self.out_num)
        # print('shape after mlp.view:', x.shape)
        return x

class CNN2(nn.Module):
    def __init__(self,in_num,out_num):
        super(CNN2, self).__init__()
        self.in_num=in_num
        self.out_num=out_num
        self.conv=nn.Sequential(
            nn.Conv1d(1,8,kernel_size=5,stride=5,padding=0),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),

            nn.Conv1d(8,32,kernel_size=4,stride=2,padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4,stride=2),

            nn.Conv1d(32,64,kernel_size=4,stride=2,padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4,stride=2)
        )
        self.MLP=nn.Sequential(
            nn.Linear(in_num,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128,out_num)
        )
    def forward(self,x):
        x=self.conv(x)
        # print("Shape after conv:", x.shape)
        x=x.view(-1,self.in_num)
        # print("Shape after conv.view:", x.shape)
        x=self.MLP(x)
        # print('shape after mlp:',x.shape)
        x=x.view(-1,self.out_num)
        # print('shape after mlp.view:', x.shape)
        return x

if __name__ == '__main__':
    #计算模型参数
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # # 创建简单的CNN模型
    # model(cnn2) = CNN(512,5)
    # # 打印模型参数数量
    # print(f"总参数数量: {count_parameters(model(cnn2))}")
    # #512  158701
    # #1024 224237
    #打印模型
    model1=CNN(512,5)
    print('model:',model1)

    #绘制网络
    dummy_input = torch.randn(1, 1, 512)
    # # 生成模型结构图，不适用
    # model(cnn2) = CNN(512, 5)
    # output = model(cnn2)(dummy_input)
    # graph = make_dot(output, params=dict(model(cnn2).named_parameters()))
    # graph.render("simple_cnn_graph")


