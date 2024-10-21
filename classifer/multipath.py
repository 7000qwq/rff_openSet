#coding=gbk
import numpy as np
import matplotlib.pyplot as plt

path1='../dataset/org_signal/me/g1_/'
path2='../dataset/multipath/3path_f2/'

# ģ��ྶ����Ӱ��
np.random.seed(20)
num_paths = 3  # �ྶ����
# delays = np.random.rand(num_paths)  # ������ɶྶ���ӳ�
# amplitudes = np.random.rand(num_paths)  # ������ɶྶ��˥������
# print('delays:',delays)
# print('amp:',amplitudes)

#5path_f1:
# delays = np.array([0, 0.25, 0.5, 0.75, 1])  # �ྶ���ӳ�
delays = np.array([0, 0.2, 0.3])  # �ྶ���ӳ�
amplitudes = np.array([1, 0.5, 0.25])  # �ྶ�����

for i in range(5):#5���豸����
    print('processing device ',i)
    path_read = path1 + 'd' + str(i + 1) + '_.bin'
    path_save = path2 + str(num_paths) + 'path_d' + str(i + 1) + '_.bin'
    path_save_test = path2 + str(num_paths) + 'path_d' + str(i + 1) + '_test.bin'
    print('read:', path_read)
    print('save:', path_save)

    original_sig=np.fromfile(path_read,dtype=np.float32)
    print('shape of org:',original_sig.shape)

    #��һ��1��д��../dataset/multipath/5/�ļ�
    norm_sig=np.divide(original_sig,np.max(original_sig))#��һ��
    original_i_signal = norm_sig[0::2]  # In-phase ͨ��
    original_q_signal = norm_sig[1::2]  # Quadrature ͨ��
    mean = np.mean(original_i_signal)
    print('mean:', np.mean(original_i_signal))

    #��һ��2��д��../dataset/multipath/5path/�ļ�
    # original_i_signal = original_sig[0::2]  # In-phase ͨ��
    # original_q_signal = original_sig[1::2]  # Quadrature ͨ��
    # original_i_signal -= np.mean(original_i_signal)
    # original_q_signal -= np.mean(original_q_signal)
    # sigma = np.sqrt(np.mean(original_i_signal * original_i_signal + original_q_signal * original_q_signal))
    # print('max:', np.max(sigma))
    # original_i_signal /= sigma
    # original_q_signal /= sigma
    # max=np.max(sigma)
    # mean=np.mean(original_i_signal)
    # print('mean:', np.mean(original_i_signal))

    # ��ʼ����Ӱ����ź�
    affected_i_signal = np.zeros_like(original_i_signal)#np.zeros([len(original_i_signal),1])
    # print(affected_i_signal.size)
    affected_q_signal = np.zeros_like(original_q_signal)#np.zeros_like([len(original_q_signal),1])
    affected_iq=np.zeros([len(original_i_signal)+len(original_q_signal),])
    # print('shapeof iq:',affected_iq.shape)
    # print("l:",len(affected_i_signal))


    # ��Ӷ�·��Ӱ�쵽 I �� Q ͨ��
    for i in range(num_paths):
        print('mean:',np.mean(original_i_signal))
        path_i_signal = amplitudes[i] * original_i_signal * np.cos(2 * np.pi * delays[i] * np.arange(len(original_i_signal)))
        path_q_signal = amplitudes[i] * original_q_signal * np.sin(2 * np.pi * delays[i] * np.arange(len(original_q_signal)))
        # print('soi:',path_q_signal.shape)
        # print('soa:',affected_i_signal.shape)
        affected_i_signal += path_i_signal
        affected_q_signal += path_q_signal

    # �������жྶЧӦ�� IQ �ź�
    affected_iq[0::2]=affected_i_signal
    affected_iq[1::2]=affected_q_signal

    affected_iq=affected_iq.astype('float32')#�õ�����float64�ģ�Ҫ����һ��ת��
    print('max:', np.max(affected_iq))
    print('dtype:', affected_iq.dtype)
    affected_iq.tofile(path_save)

    # test_part = np.zeros(([5e6, ]))
    # test_part = affected_iq[0:5e6]
    # test_part.tofile(path_save_test)



