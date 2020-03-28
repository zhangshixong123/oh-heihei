import time
import torch
from torch import nn, optim
#import data_loader as dl
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
#为了显示中文而不乱码
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#这里路径注意不能把包名包含在内
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.utils.data as Data

def evaluate_accuracy_t(t_iter, net, device=None,train=False):    
    t_l_sum, t_acc_sum, n, batch_count, start,t_acc_labels,l_accs= 0.0, 0.0, 0, 0, time.time(),torch.zeros(num_label),np.zeros(num_label)
    for X, y in t_iter:
        #一定要注意进入网络的输入数据类型，会影响网络的一系列计算（Pytorh是关于张量的计算）
        X = X.squeeze(-2)       
        output = net(X) #直接进行了forward()运算
        #注意output与y类型的匹配      
        l = losss(output, y)
        #如果是训练，则开启优化过程
        if train:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()      
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
    if train:
        return t_l_sum/batch_count,t_acc_sum/n, l_accs[0] / t_acc_labels[0].item(), l_accs[1] / t_acc_labels[0].item(), time.time() - start,net
    else:
        return t_l_sum/batch_count,t_acc_sum/n, l_accs[0] / t_acc_labels[0].item(), l_accs[1] / t_acc_labels[0].item(), time.time() - start
    

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
#C1 = C1[k:k+518]  
C2 = np.load('people_10.npy')
raw_num_C1  = int(len(C1)*0.85) 
raw_num_C2  = int(len(C2)*0.85) 
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


val_C1 = train_C1*3/17 - train_C1*3/17%batch_size
val_C2 = train_C2*3/17 - train_C2*3/17%batch_size



train_labels = [int(x) for x in train_labels]
test_labels = [int(x) for x in test_labels]

#列表数据tensor化
train_data = torch.tensor(train_inputs, dtype = torch.float32)
train_labels = torch.tensor(train_labels, dtype= torch.long)
test_data = torch.tensor(test_inputs, dtype = torch.float32)
test_labels = torch.tensor(test_labels, dtype= torch.long)
#组合数据和标签
dataset = Data.TensorDataset(train_data,train_labels)
testset = Data.TensorDataset(test_data,test_labels)

#随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle= True)
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
            nn.Dropout(0.5),
            nn.Linear(num_hiddens1,num_hiddens2), 
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(num_hiddens2,num_outputs),
        )

#定义代价函数
losss = nn.CrossEntropyLoss()#这里是二次代价函数MSELoss()，还有nn.CrossEntropyLoss()交叉熵代价函数
#定义优化器
optimizer = optim.SGD(net.parameters(), lr = eta)#Adam
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)
from torch.nn import init

#权值初始化
init.normal_(net[0].weight, mean = 0, std = 0.01)
init.normal_(net[3].weight, mean = 0, std = 0.01)
init.normal_(net[6].weight, mean = 0, std = 0.01)
init.constant_(net[0].bias, val =0)   #也可直接修改bias的data,net[0],bias.data.fill(0)  
init.constant_(net[3].bias, val =0) 
init.constant_(net[6].bias, val =0) 
    
############################################
#训练模型,网络最好在主程序中存在？
train_losses ,train_accs ,test_accs ,train_times ,test_times ,test_acc_cars ,test_acc_peoples= [],[],[],[],[],[],[]
for epoch in range(0, num_epochs):
    
    #开始训练
    train_loss,train_acc,train_acc_car,train_acc_people,train_time,net= evaluate_accuracy_t(data_iter, net,train=True)   
    #传入test_iter来检验
    test_loss,test_acc,test_acc_car,test_acc_people,test_time = evaluate_accuracy_t(test_iter, net)
    #输出识别情况
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec, test_acc_car %.4f, test_acc_people %.4f'
             % (epoch + 1, train_loss, train_acc, test_acc, train_time, test_acc_car, test_acc_people))

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    test_acc_cars.append(test_acc_car)
    test_acc_peoples.append(test_acc_people)
    train_times.append(train_time)
    test_times.append(test_time)
    
epoches = np.arange(1,num_epochs+1 ,1)

io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\000\train_losses.mat', {'train_losses': train_losses})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\000\train_accs.mat', {'train_accs': train_accs})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\000\test_accs.mat', {'test_accs': test_accs})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\000\test_acc_cars.mat', {'test_acc_cars': test_acc_cars})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\000\test_acc_peoples.mat', {'test_acc_peoples': test_acc_peoples})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\000\train_times.mat', {'train_times': train_times})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\000\test_times.mat', {'test_times': test_times})
io.savemat(r'C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\000\epoches.mat', {'epoches': epoches})

