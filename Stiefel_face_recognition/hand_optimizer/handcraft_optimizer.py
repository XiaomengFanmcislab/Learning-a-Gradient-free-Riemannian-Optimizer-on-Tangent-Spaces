import torch
import torch.nn as nn
from torch.autograd import Variable as V
from MatrixLSTM.MatrixLSTM import MatrixLSTM
from hand_optimizer.retraction import Retraction

class Hand_Optimizee_Model(nn.Module): 
    def __init__(self,lr):
        super(Hand_Optimizee_Model,self).__init__()
        self.lr=lr
        self.retraction=Retraction(self.lr)

    def forward(self,grad,M,state):
        #grad_R=grad-torch.matmul(torch.matmul(M,M.permute(0,2,1)),grad)

        #projection
        M_temp=torch.matmul(M.permute(0,2,1),grad)
        M_temp=0.5*(M_temp+M_temp.permute(0,2,1))
        M_update=grad-torch.matmul(M,M_temp)

        M = self.retraction(M,M_update)
        #grad_R=grad
        #M=M-self.lr*grad_R

        return M,state

