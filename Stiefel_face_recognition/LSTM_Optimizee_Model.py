import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

from MatrixLSTM.MatrixLSTM import MatrixLSTM
from MatrixLSTM_lr.MatrixLSTM import MatrixLSTM_lr
from torch.nn import functional
#import DataPre
#from DataPre.DataPre import Datapre



class LSTM_Optimizee_Model(nn.Module):    
    def __init__(self,opt,input_size, output_size, batchsize_data,batchsize_para):
        super(LSTM_Optimizee_Model,self).__init__()

        self.input_size=input_size
        self.output_size=output_size
        self.batchsize_data=batchsize_data
        self.batchsize_para=batchsize_para


        self.lstm=MatrixLSTM(input_size, output_size)
        self.lstm_lr=MatrixLSTM_lr(input_size, output_size)

        self.batch_size=batchsize_data//batchsize_para
        
        self.weight1=nn.Parameter(torch.randn(input_size, self.batch_size),requires_grad =True)
        self.weight2=nn.Parameter(torch.randn(117,self.output_size),requires_grad=True)

        self.conv1=nn.Conv1d(1,1,3,3,padding=0)
        self.conv2=nn.Conv1d(1,1,6,3,padding=0)

        self.pro1=nn.Parameter(torch.randn(input_size,input_size),requires_grad=True)
        self.pro2=nn.Parameter(torch.randn(input_size,input_size),requires_grad=True)


        
        self.scale=1

    
    def forward(self, input_gradients, prev_state,inputs_data,label):

        input_gradients = input_gradients.cuda()
        dim=input_gradients.shape[1]
        label=label.view(self.batchsize_data)
        #print(label.shape)
        label_one_hot=functional.one_hot(label,num_classes=38)
        #print(label_one_hot.shape)




        
        #inputs_data_N=inputs_data.shape[0]
        #print('--------------------N------------------------',inputs_data_N)
        #batch_size=self.batchsize_data//self.batchsize_para
        input_data_temp =inputs_data.view(self.batchsize_data,self.input_size)
        #input_label=label.view(self.batchsize_data,self.output_size)
        input_data_label=torch.cat((input_data_temp,label_one_hot),1)
        input_output_size=self.input_size+self.output_size
        input_temp=input_data_label.view(self.batchsize_data,1,input_output_size)
        #print(input_temp.shape)
        

        
        input_conv1=self.conv1(input_temp)
        #print(input_conv1.shape)
        input_temp_conv1=nn.functional.relu(input_conv1)
        input_conv2=self.conv2(input_temp_conv1)
        #print(input_conv2.shape)
        input_temp_conv2=nn.functional.relu(input_conv2)
        #input_temp=input_temp_conv2.view(self.batchsize_data,117)
        input_temp=input_temp_conv2.view(self.batchsize_para,self.batchsize_data//self.batchsize_para,-1)
        #print(input_temp.shape)
        input_temp=torch.matmul(self.weight1,input_temp)
        input_temp=torch.matmul(input_temp,self.weight2)
 
       

        input_gradients=input_gradients+input_temp.cuda()
        #print(input_gradients.shape)
        #print(self.pro1.shape)
        input_gradients=torch.matmul(self.pro1,input_gradients)
        input_gradients=torch.matmul(self.pro2,input_gradients)
        

        if prev_state is None: 
            prev_state = (torch.zeros(self.batchsize_para,self.input_size,self.output_size).cuda(),
                            torch.zeros(self.batchsize_para,self.input_size,self.output_size).cuda(),
                            torch.zeros(self.batchsize_para,self.input_size,self.output_size).cuda(),
                            torch.zeros(self.batchsize_para,self.input_size,self.output_size).cuda()
                            )        
        
        update_dir , next_state_dir = self.lstm(input_gradients, prev_state)
        update_lr , next_state_lr= self.lstm_lr(input_gradients, prev_state)

    
        
        
        #print('dir state',torch.sum(next_state_dir[0]),torch.sum(next_state_dir[1]),torch.sum(next_state_dir[2]),torch.sum(next_state_dir[3]))
        #print('lr state',torch.sum(next_state_lr[0]),torch.sum(next_state_lr[1]),torch.sum(next_state_lr[2]),torch.sum(next_state_lr[3]))


        next_state=( torch.mul(next_state_dir[0],next_state_lr[0]), torch.mul(next_state_dir[1],next_state_lr[1]), torch.mul(next_state_dir[2],next_state_lr[2]), torch.mul(next_state_dir[3],next_state_lr[3])  )
        #next_state=( next_state_dir[0]+next_state_lr[0], next_state_dir[1]+next_state_lr[1], next_state_dir[2]+next_state_lr[2], next_state_dir[3]+next_state_lr[3]  )
        #next_state=( next_state_dir[0], next_state_dir[1], next_state_dir[2], next_state_dir[3]  )
        
        next_state1 = (F.normalize(next_state[0].view(self.batchsize_para,self.input_size*self.output_size), p=2, dim=1)).view(self.batchsize_para,self.input_size,self.output_size)
        next_state2 = (F.normalize(next_state[1].view(self.batchsize_para,self.input_size*self.output_size), p=2, dim=1)).view(self.batchsize_para,self.input_size,self.output_size)
        next_state3 = (F.normalize(next_state[2].view(self.batchsize_para,self.input_size*self.output_size), p=2, dim=1)).view(self.batchsize_para,self.input_size,self.output_size)
        next_state4 = (F.normalize(next_state[3].view(self.batchsize_para,self.input_size*self.output_size), p=2, dim=1)).view(self.batchsize_para,self.input_size,self.output_size)

        next_state=(next_state1,next_state2,next_state3,next_state4)
        
        return update_lr, update_dir , next_state   