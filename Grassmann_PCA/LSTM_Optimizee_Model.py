import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

from MatrixLSTM.MatrixLSTM import MatrixLSTM
from MatrixLSTM_lr.MatrixLSTM import MatrixLSTM_lr
#import DataPre
#from DataPre.DataPre import Datapre



class LSTM_Optimizee_Model(nn.Module):    
    def __init__(self,opt,input_size, output_size, batchsize_data,batchsize_para):
        super(LSTM_Optimizee_Model,self).__init__()
        
        self.input_size=input_size
        self.output_size=output_size
        self.batchsize_data=batchsize_data
        self.batchsize_para=batchsize_para
        self.batch_size=batchsize_data//batchsize_para
        
        self.lstm=MatrixLSTM(input_size, output_size)
        self.lstm_lr=MatrixLSTM_lr(input_size, output_size)
        
        self.weight=nn.Parameter(torch.randn( output_size,self.batch_size),requires_grad =True)

        self.pro1=nn.Parameter(torch.randn(input_size,input_size),requires_grad=True)
        self.pro2=nn.Parameter(torch.randn(input_size,input_size),requires_grad=True)

        self.conv1=nn.Conv2d(1,1,3,1,padding=1)
        self.conv2=nn.Conv2d(1,1,5,1,padding=2)


        

        self.scale=1

    
    def forward(self, input_gradients, prev_state,inputs_data):

        input_gradients = input_gradients.cuda()
        dim=input_gradients.shape[1]

        
        inputs_data=inputs_data.permute(0,2,1)
        inputs_data=inputs_data.view(self.batchsize_data,784)
        inputs_data=inputs_data.view(self.batchsize_data,1,28,28)

        input_conv1=self.conv1(inputs_data)
        input_temp_conv1=nn.functional.relu(input_conv1)

        input_conv2=self.conv2(input_temp_conv1)
        #print('conv2_shape',input_conv2.shape)
        input_temp_conv2=nn.functional.relu(input_conv2)


        input_temp=input_temp_conv2.view(self.batchsize_para,self.batch_size,-1)
        input_temp=torch.matmul(self.weight,input_temp)



        input_temp=input_temp.permute(0,2,1)
       
   

        input_gradients=input_gradients+input_temp.cuda()
        #print(input_gradients.shape)

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