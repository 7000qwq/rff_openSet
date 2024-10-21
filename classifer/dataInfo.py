#coding=gbk
import json
import numpy as np


inputFilename = '../dataset/datainfo/neu_m046k5676'
with open("{}.sigmf-meta".format(inputFilename),'rb') as read_file:
	meta_dict = json.load(read_file)
print(meta_dict)

#############读取

# with open("{}.sigmf-data".format(inputFilename),'rb') as read_file:
# 	binary_data = read_file.read()
# fullVect = np.frombuffer(binary_data, dtype = np.complex128)	#这里的数据类型自行选择，参考信号数据集中的要求进行更改
# even = np.real(fullVect)	#提取复数信号中的实部
# odd = np.imag(fullVect)		#提取复数信号中的虚部
