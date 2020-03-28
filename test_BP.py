#利用训练好的模型去识别其他车的数据，查看识别情况
import time
import torch
from torch import nn, optim
#import data_loader as dl
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import sys
sys.path.append(r"C:\Users\xiong zhang\.spyder-py3\Code-learn\github_code_learn\BP_item\early-stopping-pytorch-master")

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

#导入保存的训练模型
test_net = torch.load('last_net.pt')
#test_net.load_state_dict(torch.load('checkpoint.pt'))
losss = nn.CrossEntropyLoss()#这里是二次代价函数MSELoss()，还有nn.CrossEntropyLoss()交叉熵代价函数
#定义优化器
optimizer = optim.SGD(test_net.parameters(), lr = 0.03)#Adam
#初始化参数
num_epochs  = 100
eta = 0.003
batch_size = 5
num_label = 2
############################################
C1 = np.load('horse_10.npy')#car_10
k=2000
#C1 = C1[k:k+518]  

num_C1 = len(C1) - len(C1)%batch_size 

train_inputs = list(C1[0:num_C1])
#用0表示车 ,1表示人   
train_labels = list(np.zeros(num_C1))
train_labels = [int(x) for x in train_labels]

#列表数据tensor化
train_data = torch.tensor(train_inputs, dtype = torch.float32)
train_labels = torch.tensor(train_labels, dtype= torch.long)

#组合数据和标签
dataset = Data.TensorDataset(train_data,train_labels)

#随机读取小批量
test_iter = Data.DataLoader(dataset, batch_size, shuffle= False)

test_loss,test_acc,test_acc_car,test_acc_people,test_time = evaluate_accuracy_t(test_iter, test_net, mod = 'test')    
