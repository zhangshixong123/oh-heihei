#import matplotlib.pyplot as plt
import numpy as np
#import pywt
#import win32ui
import struct
from scipy import io
#import sys

#初始设定
num_point = 10 #特征点数
#smp=1000;       #采样率
#frames=1;       #每帧最长时间（s）
#tpot=smp*frames  #每帧时间点

#读取文件
import tkinter  
import tkinter.filedialog
#root = tkinter.Tk()
filenames = tkinter.filedialog.askopenfilenames()
datas = []
fnum = len( filenames )
for i in range( fnum ):
    try:
        f = open(filenames[i], 'rb')
        content = f.read()
        #1个16进制数据占1个字节。这里一个float数据占4个字节，所以➗4，d_len为数据长度
        d_len = int(len(content) / 4)
        data = struct.unpack('f' * d_len, content) 
        data = list(data)
        datas.append(data)
    finally:
        if f:
            f.close()
#winrm.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0));
#先变为数组数据，方便索引
#把数据排成一排。然后再按501取
data_end = []
for data in datas:
    #i从0开始
    for i in range(int(len(data)/num_point)):
        data_end.append(data[i*num_point:(i+1)*num_point])
#先转换为列表数据，便于修改，然后再变为数组数据，用于形状的改变
A = np.array(data_end)
C = A.reshape(-1,1,num_point)
C1 = A.reshape(-1,num_point)
'''
#载入数据
B1 = np.load('car_2.npy')
B2 = np.load('people_2.npy')
#改变数组形状以适合矩阵
B1  = B1.reshape(-1,B1.shape[2])
B2  = B2.reshape(-1,B2.shape[2])
#保存为mat文件
io.savemat('B1.mat', {'B1': B1})
io.savemat('B2.mat', {'B2': B2})
#使用前需要调整输入点数
io.savemat('horse_10.mat', {'horse_10': C1})
io.savemat('people_60.mat', {'people_60': C1})
#数组的粘贴操作
ab1=np.concatenate((L1, L2),axis=2)
ab2=np.concatenate((M1, M2),axis=2)
#保存数组文件到电脑中
np.save('horse_10.npy',C)
np.save('people_60.npy',C)
np.savez('C1.npy',C1)
'''