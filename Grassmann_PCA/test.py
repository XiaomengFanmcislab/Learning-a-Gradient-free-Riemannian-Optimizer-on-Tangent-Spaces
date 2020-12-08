import torch
import torch.nn as nn
from DataSet.MNIST import MNIST
from LSTM_Optimizee_Model_test import LSTM_Optimizee_Model
import config2
import numpy as np
from utils import FastRandomIdentitySampler
from torch.autograd import Variable
opt = config2.parse_opt()

#retraction=Retraction(1)

def f(inputs,theta):
    w=theta.mm(torch.transpose(theta,0,1))
    y_pred=w.mm(inputs)
    L=(torch.norm(inputs-y_pred)).pow(2)
    return L

def retraction(weight,update_vector):
    u,s,v=torch.svd(weight-update_vector)

    #print('retraction_before',weight)
    weight=u.mm(torch.transpose(v,0,1))
    #print('retraction_after',weight)
   
    return weight


batchsize_data=int(opt.batchsize_data/opt.batchsize_para)
train_mnist = MNIST(opt.datapath, train=True)
train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=batchsize_data,shuffle=True, drop_last=True, num_workers=0)


print(batchsize_data)

LSTM_Optimizee = LSTM_Optimizee_Model(opt,opt.DIM, opt.outputDIM, batchsize_data=opt.batchsize_data, batchsize_para=opt.batchsize_para).cuda()

State_optimizer=torch.load('yourfile.pth')

LSTM_Optimizee.load_state_dict(State_optimizer,strict=False)



Epoches=7000
print(Epoches)
DIM=784
outputDIM=128
X=[]
Y=[]
X_ours=[]
Y_ours=[]
X_csgd=[]
Y_csgd=[]
iterations=0
all_iter=0
count=20
N=60000
device=torch.device("cuda:0")
data1=np.load('data/dim784_training_images_bool.npy')
data2=torch.from_numpy(data1)
data2=data2.float()
Data=data2.to(device)
Data=torch.transpose(Data,0,1)

theta0=torch.empty(DIM,outputDIM,dtype=torch.float,device=device,requires_grad=True)
torch.nn.init.orthogonal_(theta0)

theta0=Variable(theta0)
theta0.requires_grad=True

M_test=theta0
weight_after=theta0

state=(torch.zeros(DIM,outputDIM).cuda(),torch.zeros(DIM,outputDIM).cuda(),torch.zeros(DIM,outputDIM).cuda(),torch.zeros(DIM,outputDIM).cuda(),)

loss_all=f(Data,theta0)
X_ours.append(iterations)
Y_ours.append(loss_all.item()/N)
print('all_iter:{},loss_all:{}'.format(all_iter,loss_all.item()/N))

for i in range(Epoches):
    it=0
    for j, data in enumerate(train_loader, 0):
        inputs,label=data
        inputs=inputs.to(device)
        inputs_tw=torch.transpose(inputs,0,1)
        if it==0:
            loss=f(inputs_tw,theta0)
            loss.backward()
            theta_grad=theta0.grad.data
            P=theta_grad-torch.matmul(torch.matmul(theta0,torch.transpose(theta0,0,1)),theta_grad)
            P=1e-4*P
            
            it=it+1

        inputs_conv=inputs.view(opt.batchsize_para,inputs.shape[0]//opt.batchsize_para,-1)
        inputs_conv=inputs_conv.permute(0,2,1)

        
        with torch.no_grad():
            
            lr, update, state = LSTM_Optimizee(P, state,inputs_conv)
            
            lr=torch.abs(lr)
            M_update=update-torch.matmul(torch.matmul(theta0,torch.transpose(theta0,0,1)),update)
            
            
            M_update=M_update.squeeze()
            lr=torch.unsqueeze(torch.squeeze(lr),0)
          

            if i<1000:
                P=P-lr*M_update
            elif i>=1000 and i<4000:
                P=P-0.01*lr*M_update
            else:
                P=P-0.001*lr*M_update
            
            try:
                M_test=retraction(theta0,P)
            except:
                print('svd')
                    
            loss_test=f(Data,M_test)
            X_ours.append(all_iter)
            Y_ours.append(loss_test.item()/N)
            print('Train_epoch:{},inner_iteration:{},all_inner:{},loss:{}'.format(i,it,all_iter,loss_test.item()/N))
            
            all_iter=all_iter+1
            
            it=it+1

            
        if it>=1:
            break
    
    try:
        weight_after=retraction(theta0,P)
        
    except:
        weight_after=theta0
        
        print('svd')
       
    theta0=weight_after
    
    theta0=Variable(theta0)
    theta0.requires_grad=True
    
    
    if i>=Epoches:
        print('END')
        break
        
     
X=np.array(X_ours)
Y=np.array(Y_ours)

np.save('yourname',X)
np.save('yourname',Y)

            