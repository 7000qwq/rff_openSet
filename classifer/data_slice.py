#coding=gbk
import numpy as np

# ###取前N个点
# #N=13107200
# for i in range(5):
#     data = np.fromfile('../dataset/org_signal/me/g1_/d' + str(i + 1) + "_.bin", dtype=np.float32)
#     save_path='../dataset/org_signal/me/g1_/512mod/d' + str(i + 1) + "_.bin"
#     x=data[:13107200]
#     print(x.shape)
#     x = np.array(x, dtype=np.float32)
#     x.tofile(save_path)
#     print('device:',i)

#处理wsnet数据，得到512的倍数个点12000*512=6144000
# for i in range(5):
#     data=np.loadtxt('../dataset/wsnet/S_800/'+str(i+1)+'_800.txt')
#     print('dtype:',data.dtype)
#     print('shape:',data.shape)
#     data.astype('float32')
#     print('shape:', data.shape)
#     x = data[:6144000]
#     print('x_shape:',x.shape)
#     x.tofile('../dataset/wsnet/S_800_bin/'+str(i+1)+'_800_512mod.bin')

# data=np.fromfile('../dataset/wsnet/S_800_bin/1_800_512mod.bin')
# print('shape:',data.shape)
###取文件的前10e6个点
# data2=np.fromfile('../dataset/multipath/3path_f2/3path_d1_.bin',dtype='float32', count=int(2e6))
# data2.tofile('../dataset/multipath/3path_f2/3path_d1_2e6.bin')
##
for i in range(5):
    path='../dataset/psd/g1/psd(new)_d'+str(i+1)+'.txt'
    pathsave='../dataset/psd/g1/psd(new)_512mod_d'+str(i+1)+'.txt'
    data=np.loadtxt(path,max_rows=512*512*96)
    np.savetxt(pathsave,data)
    print(i)
#计算模型参数

