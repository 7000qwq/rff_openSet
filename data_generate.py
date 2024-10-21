#coding=gbk
import numpy as np
import os
import sys
import scipy.fft as fft

def ifft(x):
    K=len(x)
    x=x.conjugate()
    y=fft(x)
    y=y.conjugate()/K
    return y
NUM=512
N = NUM          #512
M = NUM          #512
beta=2           #betaΪ��ɭָ����һ��Ϊ2����Ҫ�Կ�������ЧӦʱ����Ϊ1����������Ϊ������
ifFreq=False     #����ΪTrueʱ�����������ΪƵ�����ݣ�FalseΪʱ������
device_num=5
c = N / M
slide = M * N

#���ö�ȡ���ݵ�·��
# device_path='./dataset/org_signal/me/g1_/'
device_path='./dataset/SNR/me/SNR_-9/'
# device_path='./dataset/multipath/3path_f2/'#�ྶ����
# device_path='./dataset/wsnet/S_800_bin/'#wsnet���и���
device_file=[]
for i in range(5):#device_num
    # device_file.append("signal1000_" + str(i + 1) + ".txt")
    device_file.append("-9dB_d"+str(i+1)+".bin")
    # device_file.append("3path_d"+str(i+1)+"_.bin")
    # device_file.append( str(i + 1) + "_800_512mod.bin")
    print(device_path+device_file[i])

#���ô洢·��
# save_dir = './dataset/rmt_value/SNR/25/g1/512/'
save_dir = './dataset/rmt_value/SNR/-9/'#SNR
# save_dir = './dataset/rmt_value/me_multipath/3path_f2/'#multipath
# save_dir ='./dataset/rmt_value/wsnet/S_800/512/'#wsnet

save_device = []
for i in range(5):#device_num
    save_device.append("-9dB_d" + str(i + 1) + "_" + str(N) + "_" + str(M) + ".bin")
    # save_device.append("3path_d" + str(i + 1) + "_" + str(N) + "_" + str(M) + ".bin")#multipath
    # save_device.append("wsnet_s800_" + str(i + 1) + "_" + str(N) + "_" + str(M) + ".bin")
    print(save_dir+save_device[i])


for j in range(device_num):
    dataset = []
    path=device_path+device_file[j]

    start = 0
    sigRaw = np.fromfile(path, dtype=np.float32)
    if len(sigRaw) % 2 != 0:
        print('len:', len(sigRaw))
        print('odd')
        sigRaw = np.delete(sigRaw, [-1])
    sigI = sigRaw[0::2]
    sigQ = sigRaw[1::2]
    print('len(I):', len(sigI))

    if ifFreq:
        fSig = sigI + 1j * sigQ
        tSig = ifft(fSig)
        sigI = np.real(tSig)
        sigQ = np.imag(tSig)
    # ��sigI��sigQ����Ϊʱ���I/Q����


    LCE = np.zeros([N, N]) + 1j * np.zeros([N, N])
    eigTable = []
    print("Processing " + path)
    start = 0
    print('mean sigi:',np.mean(sigI))
    print('mean sigq:', np.mean(sigQ))
    sigI -= np.mean(sigI)
    sigQ -= np.mean(sigQ)
    sigma = np.sqrt(np.mean(sigI * sigI + sigQ * sigQ))
    print('max:',np.max(sigma))
    sigI /= sigma
    sigQ /= sigma
    thh = 0
    if beta == 1:
        thh = sigma * sigma
    else:
        thh = np.mean(np.abs(sigI + 1j * sigQ))

    while ((start + M * N) < len(sigI)):
        if beta == 2:
            H = sigI[start:start + M * N] + 1j * sigQ[start:start + M * N]
            # print(len(H))
        elif beta == 1:
            H = sigI * sigI + sigQ * sigQ
        # H����һ������һά���� ֮��reshape
        # W-L�������W=HH*
        # ���ݴ�ɭϵ��beta��������� �ֱ�����Ӧ��H����

        if ((np.mean(np.abs(H))) > 0.1 * thh):
            H = H.reshape(N, M)
            print('len(H):', len(H))
            LCE = np.matmul(H, H.transpose().conjugate())
            # LCE����W����
            print('len(LCE):', len(LCE))
            eigTable = np.concatenate([eigTable, np.sort(np.real(np.linalg.eigvals(LCE)))])
            print('len(eig):', len(eigTable))
            start += slide
        else:
            start += int(slide)
        # ����ĳ����ֵ�Ż����W���������ֵ

    # eigTable /= (N * beta)
    eigTable = np.divide(eigTable, (N * beta))
    print('len(eigtable):', len(eigTable))
    # ���ȼ������W��N������ֵ����ʽ(3)�����Ƶ�������ֵ�߶����N�Ĵ�С�����ȡ�
    # Ϊ�����³���ԣ����Ƚ�����ֵ��һ��Ϊ��N��������Щ���㣬�Ϳ��Եõ�����ֵ���С�

    dataset = np.concatenate([dataset, eigTable])
    print('shape(dataset):', len(dataset))
    print(save_dir+save_device[j]+" Len= "+str(len(dataset) / N))
    dataset = np.array(dataset, dtype='float32')
    dataset.tofile(save_dir+save_device[j])
