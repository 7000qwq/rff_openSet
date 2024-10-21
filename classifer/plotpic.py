#coding=gbk
import matplotlib.pyplot as plt
import seaborn as sns
# ����
accuracy = [76.7, 83.17, 91.10,  93.69,  96.12,  97.57, 98.22, 98.22, 98.06, 98.22]
snr      = [-10,    -8,    -6,     -4,      -2,      0,     5,   10,     15,     20]

# ������ɫ����
color1 = "#038355" # ��ȸ��
color2 = "#ffc34e" # ���ջ�
color3='#6b9ac8'#������
color4='#66c163'#���һ������
color5='#3d5180'#ǳ������

# ��������
font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)

# ��ͼ
sns.set_style("whitegrid") # ���ñ�����ʽ
sns.lineplot(x=snr, y=accuracy, color=color3, linewidth=2.0, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='test accuracy')
# sns.lineplot(x=x, y=y2, color=color2, linewidth=2.0, marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='Line 2')

# ��ӱ���ͱ�ǩ
# plt.title("Title", fontweight='bold', fontsize=14)
plt.xlabel("SNR(dB)", fontsize=12)
plt.ylabel("Accuracy(%)", fontsize=12)

# ���ͼ��
plt.legend(loc='upper left', frameon=True, fontsize=10)#ͼ����������sns.lineplot��label=��

# ���ÿ̶�����ͷ�Χ
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(-10, 20)
plt.ylim(70, 100)

# ������������ʽ
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1)

plt.savefig('../results/pic/SNR_rate/snr_curve.png', dpi=300, bbox_inches='tight')
# ��ʾͼ��
plt.show()
#