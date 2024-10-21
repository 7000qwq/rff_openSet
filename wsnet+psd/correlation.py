#coding=gbk
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pearson(v1, v2):
    n = len(v1)
    #simple sums
    sum1 = sum(float(v1[i]) for i in range(n))
    sum2 = sum(float(v2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in v1])
    sum2_pow = sum([pow(v, 2.0) for v in v2])
    #sum up the products
    p_sum = sum([v1[i] * v2[i] for i in range(n)])
    #分子num，分母denominator
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

# vector1 = [2,7,18,88,157, 90,177,570]
# vector2 = [3,5,15,90,180, 88,160,580]
#
# print(pearson(vector2,vector1))



#read data
filepath='../data/phase/3/0120_6min_emd/emd.xlsx'
cor = pd.read_excel(filepath)
# 计算相关系数矩阵，包含了任意两个列间的相关系数
print('相关系数矩阵为：\n', cor.corr())

# 绘制相关性热力图
plt.subplots(figsize=(8, 8))  # 设置画面大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
xlabel=['imf1','imf2','imf3','imf4','imf5','imf6','imf7','imf8','imf9']
ylabel=['imf1','imf2','imf3','imf4','imf5','imf6','imf7','imf8','imf9']
sns.heatmap(cor.corr(), annot=True, vmax=1, square=True, cmap="Blues",xticklabels=xlabel, yticklabels=ylabel)

plt.title('相关性热力图')
# plt.show()
plt.savefig('correlation.png')#保存的话要把show注释掉，不然保存的就是空白的

