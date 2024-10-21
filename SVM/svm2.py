# coding=gbk

import operator
import pandas as pd
import torch.utils.data

from sklearn import svm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib
import sklearn




class MySet(Dataset):
    def __init__(self, path):
        super(MySet, self).__init__()
        self.path = path
        self.filename = os.listdir(self.path)

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        file_item = self.filename[item]
        filepath = os.path.join(self.path, file_item)
        data = np.fromfile(filepath, 'float32')
        label = file_item.split('_')[0]
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        label = [label]
        label = np.array(label)

        return data, label

path='../dataset/wsnet/g2'
dataset=MySet(path)
trainsize=0.7*5000
testsize=0.3*5000

torch.manual_seed(27)
train_set,test_set=torch.utils.data.random_split(dataset,[trainsize,testsize])


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






