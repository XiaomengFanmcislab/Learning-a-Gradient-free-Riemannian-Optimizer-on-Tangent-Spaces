import torch
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import time
import math, random

from ReplyBuffer import ReplayBuffer
from models.EigLayer import EigLayer
from models.EigLayer_loss import EigLayer_loss
from models.m_exp import M_Exp
from models.m_log import M_Log

from retraction import Retraction
import numpy as np
from DataSet.KYLBERG import KYLBERG
import config
from LSTM_Optimizee_Model_tangent import LSTM_Optimizee_Model
opt = config.parse_opt()

eiglayer1=EigLayer()
eiglayer_loss=EigLayer_loss()
mlog=M_Log()
mexp=M_Exp()
retraction=Retraction(1)


def f(inputs,M,label,sample_num):


    n=inputs.shape[0]
    loss=0
    for i in range(n):
        l=label[i]-1
        for j in range(sample_num):
            
            AIM=torch.mm(torch.mm(inputs[i],M[l*sample_num+j]),inputs[i])
            AIM=torch.unsqueeze(AIM,0)
            M_S,M_U=eiglayer_loss(AIM)
            M_Sl=mlog(M_S)
            AIM=torch.matmul(torch.matmul(M_U,M_Sl),M_U.permute(0,2,1))
            

            p=torch.sum(torch.sum(torch.pow(AIM,2),2),1)
            loss=loss+torch.mean(p)
    loss=loss/(n*sample_num)
    return loss



device=torch.device("cuda:0")
train_mnist = KYLBERG(opt.datapath, train=True)
train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=opt.batchsize_data,shuffle=True, drop_last=False, num_workers=0)

train_loader_all = torch.utils.data.DataLoader(
        train_mnist, batch_size=2560,shuffle=True, drop_last=False, num_workers=0)

LSTM_Optimizee = LSTM_Optimizee_Model(opt, opt.DIM, opt.DIM, opt.DIM, batchsize_data=opt.batchsize_data, batchsize_para=opt.batchsize_para).cuda()

State_optimizer=torch.load('yourfile.pth')

LSTM_Optimizee.load_state_dict(State_optimizer,strict=False)

DIM=opt.DIM
batchsize_para=opt.batchsize_para
Epoches=60
all_iter=0
X_ours=[]
Y_ours=[]

M=torch.randn(batchsize_para,DIM, DIM).cuda()
for k in range(batchsize_para):
    M[k]=torch.eye(DIM).cuda()
M=Variable(M)
M.requires_grad=True
state = (torch.zeros(batchsize_para,DIM,DIM).cuda(),torch.zeros(batchsize_para,DIM,DIM).cuda(),torch.zeros(batchsize_para,DIM,DIM).cuda(),torch.zeros(batchsize_para,DIM,DIM).cuda(),) 

while(1):
    it=0
    for j, data in enumerate(train_loader, 0):
        inputs,label=data
        inputs=inputs.to(device)
        label=label.to(device)
        inputs_shape=inputs.shape
        data_set=inputs_shape[0]
        print('data_set',data_set)
        if data_set!=opt.batchsize_data:
            break

        if it==0:
            loss = f(inputs,M,label,opt.sample_num)
            loss.backward()
            M_grad=M.grad.data
            P=torch.matmul(torch.matmul(M, (M_grad+M_grad.permute(0,2,1))/2),M)
            
            P=P*5

        with torch.no_grad():
            lr, update, state = LSTM_Optimizee(P, state, inputs)
            update_R=(update+update.permute(0,2,1))/2
            P=P-lr*update_R  

        flag,M = retraction(M,P,1)

        loss_test=f(inputs,M,label,opt.sample_num)
        X_ours.append(all_iter)
        Y_ours.append(loss_test.item())
        
        print('all_iter:{},loss:{}'.format(all_iter,loss_test.item()))
            

        all_iter=all_iter+1
        it=it+1

            
        if it>=1:
            break

    M=Variable(M)
    M.requires_grad=True
    M.retain_grad()

    if all_iter>Epoches:
        break

X_ours=np.array(X_ours)
Y_ours=np.array(Y_ours)



np.save('yourname',X_ours)
np.save('yourname',Y_ours)
