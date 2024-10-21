#coding=gbk

import operator
import pandas as pd

from sklearn import svm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset,DataLoader

import matplotlib.pyplot as plt
import matplotlib
import sklearn

#
#jy
# filename = r"D:\论文\图\a13\train0.csv"
# data = pd.read_csv(filename, header=None)
# # 必须添加header=None，否则默认把第一行数据处理成列名导致缺失
# group = np.array(data.values.tolist())
#
# filename1 = r"D:\论文\图\a13\train0_8-3.csv"
# data1 = pd.read_csv(filename1, header=None)
# # 必须添加header=None，否则默认把第一行数据处理成列名导致缺失
# group1 = np.array(data1.values.tolist())
#
# train_data = group[0:480,0:50]
# test_data = group1[480:800,0:50]
# train_label = group[0:480,50:51]
# test_label = group1[480:800,50:51]
# print(len(test_label))

class MySet(Dataset):
    def __init__(self,path):
        super(MySet, self).__init__()
        self.path=path
        self.filename=os.listdir(self.path)

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        file_item=self.filename[item]
        filepath=os.path.join(self.path,file_item)
        data=np.fromfile(filepath,'float32')
        label=file_item.split('_')[0]
        data=(data-np.min(data))/(np.max(data)-np.min(data))
        label=[label]
        label=np.array(label)

        return data,label

def load_data_from_files(directory):
    data = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            label = int(filename.split('_')[0])-1  # Extract label from the file name
            filepath = os.path.join(directory, filename)
            print('filepath:',filepath)

            data0=np.loadtxt(filepath,delimiter=',',max_rows=6196000)
            print('shape:',data0.shape)
            data.append(data0)
            labels.append(label)

    data=np.array(data)
    data = data.reshape(5, -1)
    # data=np.transpose(data)
    print('data.shape:', data.shape)
    print('shape_d:',data.size)
    labels=np.array(labels)
    labels=labels.reshape(-1,1)

    return data, np.array(labels)

# def createData(path,devicrNum):
#     data=[]
#     labels=[]
#     filename=os.listdir(path)
#     L=len(filename)
#     for i in range(L):
#         filepath=path+filename[i]
#         signal=np.fromfile(filepath,'float32')
#         label=filename[i].split('_')[0]
#         labels.append(label)



# Assuming your files are in the 'data' directory
data_directory = '../dataset/wsnet/S_800/'
X, y = load_data_from_files(data_directory)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #设置分类器
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# #模型评估
# y_pred = clf.predict(X_test)
# # Evaluate accuracy or other metrics
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# 3.训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # ovr:一对多策略
classifier.fit(X_train, y_train.ravel())  # ravel函数在降维时默认是行序优先

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(X_train, y_train))
print("测试集：", classifier.score(X_test, y_test))
# 也可直接调用accuracy_score方法计算准确率

tra_label = classifier.predict(y_train)  # 训练集的预测标签
tes_label = classifier.predict(y_test)   # 测试集的预测标签
print("训练集：", accuracy_score(y_train, tra_label))
print("测试集：", accuracy_score(y_test, tes_label))

# 查看决策函数
print('train_decision_function:\n', classifier.decision_function(X_train))  # (90,3)
print('predict_result:\n', classifier.predict(X_train))


#################
#coding=gbk

import operator
import pandas as pd

from sklearn import svm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset,DataLoader

import matplotlib.pyplot as plt
import matplotlib
import sklearn

#
#jy
# filename = r"D:\论文\图\a13\train0.csv"
# data = pd.read_csv(filename, header=None)
# # 必须添加header=None，否则默认把第一行数据处理成列名导致缺失
# group = np.array(data.values.tolist())
#
# filename1 = r"D:\论文\图\a13\train0_8-3.csv"
# data1 = pd.read_csv(filename1, header=None)
# # 必须添加header=None，否则默认把第一行数据处理成列名导致缺失
# group1 = np.array(data1.values.tolist())
#
# train_data = group[0:480,0:50]
# test_data = group1[480:800,0:50]
# train_label = group[0:480,50:51]
# test_label = group1[480:800,50:51]
# print(len(test_label))

class MySet(Dataset):
    def __init__(self,path):
        super(MySet, self).__init__()
        self.path=path
        self.filename=os.listdir(self.path)

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        file_item=self.filename[item]
        filepath=os.path.join(self.path,file_item)
        data=np.fromfile(filepath,'float32')
        label=file_item.split('_')[0]
        data=(data-np.min(data))/(np.max(data)-np.min(data))
        label=[label]
        label=np.array(label)

        return data,label

def load_data_from_files(directory):
    data = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            label = int(filename.split('_')[0])-1  # Extract label from the file name
            filepath = os.path.join(directory, filename)
            print('filepath:',filepath)

            data0=np.loadtxt(filepath,delimiter=',',max_rows=6196000)
            print('shape:',data0.shape)
            data.append(data0)
            labels.append(label)

    data=np.array(data)
    data = data.reshape(5, -1)
    # data=np.transpose(data)
    print('data.shape:', data.shape)
    print('shape_d:',data.size)
    labels=np.array(labels)
    labels=labels.reshape(-1,1)
    print('labelshape:',labels.shape)

    return data, np.array(labels)

def createData(path,devicrNum):
    data=[]
    labels=[]
    filename=os.listdir(path)
    L=len(filename)
    for i in range(L):
        filepath=path+'/'+filename[i]
        signal=np.fromfile(filepath,'float32')
        label=filename[i].split('_')[0]
        labels.append(label)
        data.append(signal)
    # data=np.array(data)
    # data=data.reshape(devicrNum,-1)
    # label=np.array(label)
    # label=np.reshape()
    return np.array(data),np.array(labels)



# Assuming your files are in the 'data' directory
# data_directory = '../S_800/'

data_directory = '../SNR/-10/'
X, y = createData(data_directory,5)

# data_directory='../SNR/5'
# data=MySet(data_directory)

y=y.reshape(-1,1)
print('data:',y.shape)
print('x',X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #设置分类器
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# #模型评估
# y_pred = clf.predict(X_test)
# # Evaluate accuracy or other metrics
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# 3.训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # ovr:一对多策略
classifier.fit(X_train, y_train.ravel())  # ravel函数在降维时默认是行序优先

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(X_train, y_train))
print("测试集：", classifier.score(X_test, y_test))
# 也可直接调用accuracy_score方法计算准确率

tra_label = classifier.predict(y_train)  # 训练集的预测标签
tes_label = classifier.predict(y_test)   # 测试集的预测标签
print("训练集：", accuracy_score(y_train, tra_label))
print("测试集：", accuracy_score(y_test, tes_label))

# 查看决策函数
print('train_decision_function:\n', classifier.decision_function(X_train))  # (90,3)
print('predict_result:\n', classifier.predict(X_train))


#5 0.888775 0.874
