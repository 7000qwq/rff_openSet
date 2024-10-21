#coding=gbk
import matplotlib.pyplot as plt
import seaborn as sns
# 数据
accuracy = [76.7, 83.17, 91.10,  93.69,  96.12,  97.57, 98.22, 98.22, 98.06, 98.22]
snr      = [-10,    -8,    -6,     -4,      -2,      0,     5,   10,     15,     20]

# 设置颜色代码
color1 = "#038355" # 孔雀绿
color2 = "#ffc34e" # 向日黄
color3='#6b9ac8'#竹月蓝
color4='#66c163'#左边一样的绿
color5='#3d5180'#浅海昌蓝

# 设置字体
font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)

# 绘图
sns.set_style("whitegrid") # 设置背景样式
sns.lineplot(x=snr, y=accuracy, color=color3, linewidth=2.0, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='test accuracy')
# sns.lineplot(x=x, y=y2, color=color2, linewidth=2.0, marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='Line 2')

# 添加标题和标签
# plt.title("Title", fontweight='bold', fontsize=14)
plt.xlabel("SNR(dB)", fontsize=12)
plt.ylabel("Accuracy(%)", fontsize=12)

# 添加图例
plt.legend(loc='upper left', frameon=True, fontsize=10)#图例的名字在sns.lineplot的label=里

# 设置刻度字体和范围
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(-10, 20)
plt.ylim(70, 100)

# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1)

plt.savefig('../results/pic/SNR_rate/snr_curve.png', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()
#