#coding=gbk
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成一些示例数据（你应该用你自己的数据替换这部分）
# 假设有两个特征（X1和X2），以及相应的标签（y）
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 5, size=100)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建多分类SVM模型
clf = svm.SVC(kernel='linear')

# 在训练集上训练模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 注意：在多分类问题中，准确度可能不是唯一的性能指标，你可能还需要关注其他指标，例如混淆矩阵、分类报告等。

# 可视化决策边界
import matplotlib.pyplot as plt

# 绘制训练集和测试集的散点图
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, marker='x')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格以绘制决策边界
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制等高线
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.title('Multiclass SVM Classification')
plt.show()
