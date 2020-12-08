import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

from MatrixLSTM.MatrixLSTM import MatrixLSTM
from MatrixLSTM_lr.MatrixLSTM import MatrixLSTM_lr


class LSTM_Optimizee_Model(nn.Module):    
    def __init__(self,opt,input_size, hidden_size, output_size, batchsize_data,batchsize_para):
        super(LSTM_Optimizee_Model,self).__init__()
        self.lstm=MatrixLSTM(input_size, hidden_size, output_size)
        self.lstm_lr=MatrixLSTM_lr(input_size, hidden_size, output_size)


        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.batchsize_data=batchsize_data
        self.batchsize_para=batchsize_para

        self.batch_size=batchsize_data//batchsize_para

        self.conv1=nn.Conv2d(1,1,1,1,padding=0)
        self.conv2=nn.Conv2d(1,1,3,1,padding=1)

        self.weight1=nn.Parameter(torch.randn(1,self.batch_size),requires_grad=True)
        self.pro1=nn.Parameter(torch.randn(input_size,input_size),requires_grad=True)
        self.pro2=nn.Parameter(torch.randn(input_size,input_size),requires_grad=True)
        self.device=torch.device("cuda:0")
        
        self.scale=1

    
    def forward(self, input_gradients, prev_state,inputs_data):

        input_gradients = input_gradients.cuda()
        dim=input_gradients.shape[1]

        inputs_data=inputs_data.view(self.batchsize_data,1,5,5)
        input_conv1=self.conv1(inputs_data)
        #print('conv1_shape',input_conv1.shape)
        input_temp_conv1=nn.functional.relu(input_conv1)

        input_conv2=self.conv2(input_temp_conv1)
        #print('conv2_shape',input_conv2.shape)
        input_temp_conv2=nn.functional.relu(input_conv2)


        input_temp=input_temp_conv2.view(self.batchsize_para,self.batch_size,-1)

        #print(self.weight1.shape)
        #print(input_temp.shape)

        input_temp=torch.matmul(self.weight1,input_temp)
        #print(input_temp.shape)
        input_temp=input_temp.view(self.batchsize_para,5,5)
        input_gradients_temp=input_gradients+input_temp.to(self.device)

        input_gradients_temp=torch.matmul(torch.matmul(torch.transpose(self.pro1,0,1),input_gradients_temp),self.pro1)
        input_gradients_temp=torch.matmul(torch.matmul(torch.transpose(self.pro2,0,1),input_gradients_temp),self.pro2)

        

        if prev_state is None: 
            prev_state = (torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            #torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda(),
                            #torch.zeros(self.batchsize_para,self.hidden_size,self.hidden_size).cuda()
                            )        
        
        update_dir , next_state_dir = self.lstm(input_gradients_temp, prev_state)
        update_lr , next_state_lr= self.lstm_lr(input_gradients_temp, prev_state)


        #next_state=( torch.mul(next_state_dir[0],next_state_lr[0]), torch.mul(next_state_dir[1],next_state_lr[1]), torch.mul(next_state_dir[2],next_state_lr[2]), torch.mul(next_state_dir[3],next_state_lr[3])  )

        next_state=( torch.mul(next_state_dir[0],next_state_lr[0]), torch.mul(next_state_dir[1],next_state_lr[1]), torch.mul(next_state_dir[2],next_state_lr[2]), torch.mul(next_state_dir[3],next_state_lr[3])  )
        #next_state=( next_state_dir[0]+next_state_lr[0], next_state_dir[1]+next_state_lr[1], next_state_dir[2]+next_state_lr[2], next_state_dir[3]+next_state_lr[3]  )
        #next_state=( next_state_dir[0], next_state_dir[1], next_state_dir[2], next_state_dir[3]  )
        
        next_state1 = (F.normalize(next_state[0].view(self.batchsize_para,self.input_size*self.output_size), p=2, dim=1)).view(self.batchsize_para,self.input_size,self.output_size)
        next_state2 = (F.normalize(next_state[1].view(self.batchsize_para,self.input_size*self.output_size), p=2, dim=1)).view(self.batchsize_para,self.input_size,self.output_size)
        next_state3 = (F.normalize(next_state[2].view(self.batchsize_para,self.input_size*self.output_size), p=2, dim=1)).view(self.batchsize_para,self.input_size,self.output_size)
        next_state4 = (F.normalize(next_state[3].view(self.batchsize_para,self.input_size*self.output_size), p=2, dim=1)).view(self.batchsize_para,self.input_size,self.output_size)

        next_state=(next_state1,next_state2,next_state3,next_state4)
        
        
        return update_lr, update_dir , next_state   