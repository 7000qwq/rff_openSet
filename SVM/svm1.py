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
# filename = r"D:\����\ͼ\a13\train0.csv"
# data = pd.read_csv(filename, header=None)
# # �������header=None������Ĭ�ϰѵ�һ�����ݴ������������ȱʧ
# group = np.array(data.values.tolist())
#
# filename1 = r"D:\����\ͼ\a13\train0_8-3.csv"
# data1 = pd.read_csv(filename1, header=None)
# # �������header=None������Ĭ�ϰѵ�һ�����ݴ������������ȱʧ
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

# #���÷�����
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# #ģ������
# y_pred = clf.predict(X_test)
# # Evaluate accuracy or other metrics
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# 3.ѵ��svm������
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # ovr:һ�Զ����
classifier.fit(X_train, y_train.ravel())  # ravel�����ڽ�άʱĬ������������

# 4.����svc��������׼ȷ��
print("ѵ������", classifier.score(X_train, y_train))
print("���Լ���", classifier.score(X_test, y_test))
# Ҳ��ֱ�ӵ���accuracy_score��������׼ȷ��

tra_label = classifier.predict(y_train)  # ѵ������Ԥ���ǩ
tes_label = classifier.predict(y_test)   # ���Լ���Ԥ���ǩ
print("ѵ������", accuracy_score(y_train, tra_label))
print("���Լ���", accuracy_score(y_test, tes_label))

# �鿴���ߺ���
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
# filename = r"D:\����\ͼ\a13\train0.csv"
# data = pd.read_csv(filename, header=None)
# # �������header=None������Ĭ�ϰѵ�һ�����ݴ������������ȱʧ
# group = np.array(data.values.tolist())
#
# filename1 = r"D:\����\ͼ\a13\train0_8-3.csv"
# data1 = pd.read_csv(filename1, header=None)
# # �������header=None������Ĭ�ϰѵ�һ�����ݴ������������ȱʧ
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

# #���÷�����
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# #ģ������
# y_pred = clf.predict(X_test)
# # Evaluate accuracy or other metrics
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# 3.ѵ��svm������
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # ovr:һ�Զ����
classifier.fit(X_train, y_train.ravel())  # ravel�����ڽ�άʱĬ������������

# 4.����svc��������׼ȷ��
print("ѵ������", classifier.score(X_train, y_train))
print("���Լ���", classifier.score(X_test, y_test))
# Ҳ��ֱ�ӵ���accuracy_score��������׼ȷ��

tra_label = classifier.predict(y_train)  # ѵ������Ԥ���ǩ
tes_label = classifier.predict(y_test)   # ���Լ���Ԥ���ǩ
print("ѵ������", accuracy_score(y_train, tra_label))
print("���Լ���", accuracy_score(y_test, tes_label))

# �鿴���ߺ���
print('train_decision_function:\n', classifier.decision_function(X_train))  # (90,3)
print('predict_result:\n', classifier.predict(X_train))


#5 0.888775 0.874
