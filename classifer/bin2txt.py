#coding:gbk
import numpy as np
import matplotlib.pyplot as plt

save_path='../dataset/org_signal/me/g1/'
read_path='../dataset/org_signal/me/g1/'

for i in range (5):
    file_read=read_path+'d'+str(i+1)+'.bin'
    file_save=save_path+'d'+str(i+1)+'.txt'
    # print('read:',file_read)
    # print('save:',file_save)

    samples1 = np.fromfile(file_read, np.float32)
    print(samples1.size)
    np.savetxt(file_save, samples1)