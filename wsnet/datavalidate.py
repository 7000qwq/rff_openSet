import numpy as np

import os

rootpath='../dataset/org_signal/me/g2/slice/d'

rootpath='../dataset/wsnet/g1/SNR_0'
for i in range(5):
    path=rootpath+str(i+1)+'/'
    filename=os.listdir(path)
    print(f'====================device{i}========================')
    for j in range(len(filename)):
        filepath=path+filename[j]
        data=np.fromfile(filepath,'float32')
        if len(data)<4090
            filenum=filename[j].split('.')[0].split('_')[-1]
            print(f'{filenum},len{len(data)}')
