import torch
import torch.nn as nn
import math
from DataSet.YaleB import YaleB
from LSTM_Optimizee_Model import LSTM_Optimizee_Model

import config
import numpy as np
from utils import FastRandomIdentitySampler
from torch.autograd import Variable
opt = config.parse_opt()

#retraction=Retraction(1)

def nll_loss(data,label):
    label=label.item()
    n=data.shape[0]
    L=0
    for i in range(n):
        L=L+torch.abs(data[i][label[i]])
    return L


def f(inputs,target,M):
    
    loss_function=torch.nn.CrossEntropyLoss(reduction='sum')
    
    X=torch.matmul(inputs,M)
    target=target.squeeze()
    L=loss_function(X,target)
    
    return L


def retraction(inputs, grad,lr):

    
    P=-lr*grad
    PV=inputs+P
    temp_mul=torch.transpose(PV,0,1).mm(PV)
    e_0,v_0=torch.symeig(temp_mul,eigenvectors=True)
    e_0=abs(e_0)+1e-6
    
    e_0=e_0.pow(-0.5)
    temp1=v_0.mm(torch.diag(e_0)).mm(torch.transpose(v_0,0,1))
    temp=PV.mm(temp1)

    return temp

def par_projection(M,update):

    M_temp=torch.matmul(torch.transpose(M,0,1),update)
    M_temp=0.5*(M_temp+torch.transpose(M_temp,0,1))
    M_update=update-torch.matmul(M,M_temp)

    return M_update




batchsize_data=64
YaleB= YaleB(opt.datapath, train=True)
train_loader = torch.utils.data.DataLoader(
        YaleB, batch_size=batchsize_data,shuffle=True, drop_last=True, num_workers=0)


print(batchsize_data)

LSTM_Optimizee = LSTM_Optimizee_Model(opt,opt.DIM, opt.outputDIM, batchsize_data=opt.batchsize_data, batchsize_para=opt.batchsize_para).cuda()

State_optimizer=torch.load('yourfile.pth')

LSTM_Optimizee.load_state_dict(State_optimizer,strict=False)



Epoches=8000
print(Epoches)
DIM=1024
outputDIM=38
X=[]
Y=[]
X_ours=[]
Y_ours=[]
X_csgd=[]
Y_csgd=[]
iterations=0
all_iter=0
count=20
N=1900
device=torch.device("cuda:0")
data1=np.load('data/YaleB_train_3232.npy')
data2=torch.from_numpy(data1)
data2=data2.float()
Data=data2.to(device)



LABELS=np.load('data/YaleB_train_gnd.npy')
LABELS=torch.from_numpy(LABELS)
LABELS=LABELS.long()
LABELS=LABELS-1

LABELS=LABELS.to(torch.device("cuda:0"))
LABELS=LABELS.squeeze()

learning_rate=1e-6

#you can change the initial weigth by torch.randn() and nn.init.orthogonal_()
w0=np.load('w1.npy')
w1=torch.from_numpy(w0)
w1=w1.float()
w1=w1.to(device)

#theta0=torch.empty(DIM,outputDIM,dtype=torch.float,device=device,requires_grad=True)
theta0=w1

theta0=Variable(theta0)
theta0.requires_grad=True
M_csgd=theta0
state=(torch.zeros(DIM,outputDIM).cuda(),torch.zeros(DIM,outputDIM).cuda(),torch.zeros(DIM,outputDIM).cuda(),torch.zeros(DIM,outputDIM).cuda(),)

loss_all=f(Data,LABELS,theta0)
X_ours.append(all_iter)
Y_ours.append(loss_all.item()/N)

for i in range(Epoches):
    it=0
    for j, data in enumerate(train_loader, 0):
        inputs,label=data
        inputs=inputs.to(device)
        label=label.to(device)
        
        if it==0:
            loss=f(inputs,label,theta0)
            loss.backward()
            theta_grad=theta0.grad.data
            #projection
            P=par_projection(theta0,theta_grad)
            P=learning_rate*P
            
        inputs_conv=inputs.view(opt.batchsize_para,inputs.shape[0]//opt.batchsize_para,-1)
        
        
        with torch.no_grad():
            
            lr, update, state = LSTM_Optimizee(P, state,inputs_conv,label)
           
            update=update.squeeze()
            lr=torch.abs(lr)
            #projection
            M_update=par_projection(theta0,update)
            
            M_update=M_update.squeeze()
            lr=torch.unsqueeze(torch.squeeze(lr),0)

            
            if all_iter>2500:
                P=P-0.1*lr*M_update
            else:
                P=P-lr*M_update

            

            M_test=retraction(theta0,P,1)
           
            
            
            loss_test=f(Data,LABELS,M_test)
            
            X_ours.append(all_iter)
            Y_ours.append(loss_test.item()/N)
           
            print('all_iter:{},loss:{}'.format(all_iter,loss_test.item()/N))
            

            all_iter=all_iter+1

            
            it=it+1

            
        if it>=1:
            break
    
    weight_after=retraction(theta0,P,1)
    
    theta0=weight_after
    
    theta0=Variable(theta0)
    theta0.requires_grad=True

    if i>=Epoches:
        print('END')
        break
        
        
X_ours=np.array(X_ours)
Y_ours=np.array(Y_ours)

np.save('yourname',X_ours)
np.save('yourname',Y_ours)


            