#coding=gbk
import json
import numpy as np


inputFilename = '../dataset/datainfo/neu_m046k5676'
with open("{}.sigmf-meta".format(inputFilename),'rb') as read_file:
	meta_dict = json.load(read_file)
print(meta_dict)

#############��ȡ

# with open("{}.sigmf-data".format(inputFilename),'rb') as read_file:
# 	binary_data = read_file.read()
# fullVect = np.frombuffer(binary_data, dtype = np.complex128)	#�����������������ѡ�񣬲ο��ź����ݼ��е�Ҫ����и���
# even = np.real(fullVect)	#��ȡ�����ź��е�ʵ��
# odd = np.imag(fullVect)		#��ȡ�����ź��е��鲿
