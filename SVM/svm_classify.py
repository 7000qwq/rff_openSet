#coding=gbk
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ����һЩʾ�����ݣ���Ӧ�������Լ��������滻�ⲿ�֣�
# ����������������X1��X2�����Լ���Ӧ�ı�ǩ��y��
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 5, size=100)

# �����ݷ�Ϊѵ�����Ͳ��Լ�
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ���������SVMģ��
clf = svm.SVC(kernel='linear')

# ��ѵ������ѵ��ģ��
clf.fit(X_train, y_train)

# �ڲ��Լ��Ͻ���Ԥ��
y_pred = clf.predict(X_test)

# ����׼ȷ��
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# ע�⣺�ڶ���������У�׼ȷ�ȿ��ܲ���Ψһ������ָ�꣬����ܻ���Ҫ��ע����ָ�꣬����������󡢷��౨��ȡ�

# ���ӻ����߽߱�
import matplotlib.pyplot as plt

# ����ѵ�����Ͳ��Լ���ɢ��ͼ
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, marker='x')

# ���ƾ��߽߱�
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# ���������Ի��ƾ��߽߱�
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ���Ƶȸ���
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.title('Multiclass SVM Classification')
plt.show()
