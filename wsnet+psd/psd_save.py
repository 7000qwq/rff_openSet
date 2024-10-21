#coding=gbk
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

for i in range(1,2):
    for j in range(1,3):
        pathread='../dataset/pure_signal/group_a'+str(i)+'/device'+str(j)+'_a'+str(i)+'_signal.txt'
        pathsave='../dataset/psd/g'+str(i)+'/psd(new)_d'+str(j)+'.txt'
        data=np.loadtxt(pathread)

        num_fft=len(data)
        Y=fft(data,num_fft)
        Y = np.abs(Y)
        ps = 20 * np.log10(Y ** 2 / num_fft)

        np.savetxt(pathsave,ps,fmt='%.6f')#保留6个小数位
        print(f'len g{i}_{j}: {len(ps)}')

        fig=plt.figure()
        plt.plot(ps)
        plt.savefig('./psdpic/g'+str(i)+'_'+str(j)+'.png')

        # psr = ','.join(str(n) for n in ps)  # use ',' to join the string in ps
        # f = open(pathsave, 'w')
        # f.write(psr)
    print('processing : ' + str(i))


# for i in range(1,4):
#     for j in range(1,6):
#         psd=np.loadtxt('../dataset/psd/g'+str(i)+'/psd_d'+str(j)+'.txt')
#         print(f'len_g{i}_{j}:{len(psd)}')



# from scipy.signal import welch
# def get_psd_values(data, N, f_s):
#     f_values, psd_values = welch(data, fs=f_s)
#     return f_values, psd_values


# for i in range(1,2):
#     for j in range(20):
#         #data = np.loadtxt("D:\gkc_project\\data\\data_2048\\" + str(i)+"_"+str(j)+".txt")
#         data = np.loadtxt("D:\\1.RFF\\data_test1\\data_2048\\" + str(i) + "_" + str(j) + ".txt")
#         #采样点数
#         num_fft = len(data);
#         Y = fft(data, num_fft)
#         Y = np.abs(Y)
#         ps = 20 * np.log10( Y**2 / num_fft)
#         psr = ','.join(str(n) for n in ps)#use ',' to join the string in ps
#         txtName = str(i) + "_" + str(j)
#         #desktop_path = "D:\gkc_project\\data\\psd\\"  # 新创建的txt文件的存放路径
#         desktop_path = "D:\\1.RFF\data_test1\psd111\\"
#         full_path = desktop_path + txtName + '.txt'
#         f = open(full_path, 'w')
#         f.write(psr)
#     print('processing : '+ str(i))
#     ##print('data_len:',num_fft)
#
#     plt.plot(ps)
#     plt.show()


