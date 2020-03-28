import time
import torch
from torch import nn, optim
#import data_loader as dl
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import sys
sys.path.append(r"C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\early-stopping-pytorch-master")
from pytorchtools import EarlyStopping
#为了显示中文而不乱码
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#这里路径注意不能把包名包含在内
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.utils.data as Data

def evaluate_accuracy_t(t_iter, net, device=None,mod='train'):    
    t_l_sum, t_acc_sum, n, batch_count, start,t_acc_labels,l_accs= 0.0, 0.0, 0, 0, time.time(),torch.zeros(num_label),np.zeros(num_label)

    for X, y in t_iter:
        #一定要注意进入网络的输入数据类型，会影响网络的一系列计算（Pytorh是关于张量的计算）
        X = X.squeeze(-2)       
        output = net(X) #直接进行了forward()运算
        #注意output与y类型的匹配      
        l = losss(output, y)
         #如果是训练，则开启优化过程  
        if mod == 'train':
            net.train()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        else:
            net.eval() #针对dropout层和BN层
        t_l_sum += l.cpu().item()
        #取输出的10个值里的最大值，注意维度  
        x = output.argmax(dim=-1)    
        t_acc_sum += (x == y).sum().cpu().item()
        #统计各个标签出现的个数
        t_acc_label = torch.tensor([i for i in torch.bincount(y, minlength = num_label)]) 
        t_acc_labels = [t_acc_labels[i] + t_acc_label[i] for i in range(num_label)]
        #统计各标签下的正确个数
        l_acc= np.zeros(num_label)
        for i in range(batch_size):
            if x[i] == y[i]:
                l_acc[x[i].item()] +=1
        l_accs = [l_accs[i] + l_acc[i] for i in range(num_label)]     
        
        n += y.shape[0]
        batch_count += 1
       
    #返回损失，正确率，车的正确率，人的正确率，所花费时间，(以及网络)
    if mod =='train':
        return t_l_sum/batch_count,t_acc_sum/n, l_accs[0] / t_acc_labels[0].item(), l_accs[1] / t_acc_labels[1].item(), time.time() - start,net
    else:
        return t_l_sum/batch_count,t_acc_sum/n, l_accs[0] / t_acc_labels[0].item(), l_accs[1] / t_acc_labels[1].item(), time.time() - start
    

#############################################
#初始化参数
num_epochs  = 100
eta = 0.003
batch_size = 5
num_label = 2
k=0
############################################
#读入数据 C1为车，C2为人
#train_images = C1[0:100]+C2[0:600]  
#test_images  = C1[100:121]+C2[600:695]
C1 = np.load('car_10.npy')
C1 = C1[k:k+518]  
C2 = np.load('people_10.npy')
raw_num_C1  = int(len(C1)*0.8) 
raw_num_C2  = int(len(C2)*0.8) 
#将数组转换未列表，使用列表的合并功能
train_C1 = raw_num_C1 - raw_num_C1%batch_size 
train_C2 = raw_num_C2 - raw_num_C2%batch_size 
#将数据的80%作为训练集
num_C1 = len(C1) - len(C1)%batch_size 
num_C2 = len(C2) - len(C2)%batch_size 

train_inputs = list(C1[0:train_C1])+list(C2[0:train_C2])  
test_inputs  = list(C1[train_C1:num_C1])+list(C2[train_C2:num_C2])    
#用0表示车 ,1表示人     
train_labels = list(np.zeros(train_C1))+list(np.ones(train_C2))
test_labels  = list(np.zeros(num_C1-train_C1))+list(np.ones(num_C2-train_C2))

train_labels = [int(x) for x in train_labels]
test_labels = [int(x) for x in test_labels]
#只需找到大概15%用于验证的数据，且能被batch_size整除即可
'''
val_C1 = trains_C1*3/17 - trains_C1*3/17%batch_size
val_C2 = trains_C2*3/17 - trains_C2*3/17%batch_size 
val_num = val_C1 + val_C2
'''
#列表数据tensor化
train_data = torch.tensor(train_inputs, dtype = torch.float32)
train_labels = torch.tensor(train_labels, dtype= torch.long)
test_data = torch.tensor(test_inputs, dtype = torch.float32)
test_labels = torch.tensor(test_labels, dtype= torch.long)
#组合数据和标签
dataset = Data.TensorDataset(train_data,train_labels)
testset = Data.TensorDataset(test_data,test_labels)

#这里开始在85%的数据中随机选取一部分用于验证
val_num = int( len(train_inputs)*0.15 - len(train_inputs)*0.15%batch_size )
dataset, valset = torch.utils.data.random_split( dataset, [len(train_inputs) - val_num,val_num] )

#随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle= True)
val_iter = Data.DataLoader(valset, batch_size, shuffle= True)
test_iter = Data.DataLoader(testset, batch_size, shuffle= True)

#############################################
#初始化网络
num_inputs,num_hiddens1,num_hiddens2,num_outputs = C1.shape[2],30,10,2 
#from collections import OrderedDict
#定义网络模型
#net = nn.Sequential(OrderedDict( [('linear', nn.Linear(num_inputs, num_hiddens)),'linear',] ))
net = nn.Sequential(
            nn.Linear(num_inputs,num_hiddens1), 
            nn.Sigmoid(),
            #nn.Dropout(0.5),  #样本过少时可不要
            nn.Linear(num_hiddens1,num_hiddens2), 
            nn.Sigmoid(),
            #nn.Dropout(0.5),
            nn.Linear(num_hiddens2,num_outputs),
        )

#定义代价函数
losss = nn.CrossEntropyLoss()#这里是二次代价函数MSELoss()，还有nn.CrossEntropyLoss()交叉熵代价函数
#定义优化器
optimizer = optim.SGD(net.parameters(), lr = eta)#Adam
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)
#这里patience是包含样本个数的吗？
patience = 30	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)	# 关于 EarlyStopping 的代码可先看博客后面的内容

from torch.nn import init

#权值初始化
init.normal_(net[0].weight, mean = 0, std = 0.01)
init.normal_(net[2].weight, mean = 0, std = 0.01)
init.normal_(net[4].weight, mean = 0, std = 0.01)
init.constant_(net[0].bias, val =0)   #也可直接修改bias的data,net[0]。bias.data.fill(0)  
init.constant_(net[2].bias, val =0) 
init.constant_(net[4].bias, val =0) 
    
############################################
#训练模型,网络最好在主程序中存在？
train_losses, train_accs, train_acc_cars, train_acc_peoples, train_times = [],[],[],[],[]
val_losses, val_accs, val_acc_cars, val_acc_peoples, val_times = [],[],[],[],[]

for epoch in range(0, num_epochs):
    
    #开始训练
    train_loss,train_acc,train_acc_car,train_acc_people,train_time,net= evaluate_accuracy_t(data_iter, net,mod ='train')
    #训练完一个周期带入验证集查看模型性能
    val_loss,val_acc,val_acc_car,val_acc_people,val_time= evaluate_accuracy_t(val_iter, net,mod='validation') 
    #如若提前停止生效，则应跳出迭代周期，停止训练并保留    
    early_stopping(val_loss, net)
    if early_stopping.early_stop:
        torch.save({'epoch': epoch,
                    'model.state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss' : val_loss
                    }, 'checkpoint2.pt')   
        #下次尝试用以下方法导入网络
        net.load_state_dict(torch.load('checkpoint.pt'))
            
        break       #这个break后跳出一个循环
        
    #用于结果的一部分可视化
    print('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec, val_acc_car %.4f, val_acc_people %.4f'
             % (epoch + 1, train_loss, train_acc, val_acc, val_time, val_acc_car, val_acc_people))
    
    #保存每个周期训练结果
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_acc_cars.append(train_acc_car)
    train_acc_peoples.append(train_acc_people)
    train_times.append(train_time)
    #保存每个周期验证结果
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_acc_cars.append(val_acc_car)
    val_acc_peoples.append(val_acc_people)
    val_times.append(val_time)
        
#导入提前停止的模型
#net = torch.load('last_net.pt')
#传入test_iter计算识别结果,可直接在工作区看到结果
test_loss,test_acc,test_acc_car,test_acc_people,test_time = evaluate_accuracy_t(test_iter, net, mod = 'test')    
epoches = np.arange(1,num_epochs+1 ,1)
#保存模型为pt，方便其他程序导入
torch.save(net,'last_net.pt')
#保存每个周期训练结果
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\train_losses.mat', {'train_losses': train_losses})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\train_accs.mat', {'train_accs': train_accs})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\train_acc_cars.mat', {' train_acc_cars': train_acc_cars})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\train_acc_peoples.mat', {'train_acc_peoples': train_acc_peoples})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\train_times.mat', {'train_times': train_times})
#保存每个周期验证结果
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\val_losses.mat', {'val_losses': val_losses})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\train_accs.mat', {'train_accs': train_accs})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\val_acc_cars.mat', {'val_acc_cars': val_acc_cars})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\val_acc_peoples.mat', {'val_acc_peoples': val_acc_peoples})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\val_times.mat', {'val_times': val_times})
#保存迭代周期向量
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\001\epoches.mat', {'epoches': epoches})


