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
        grad_R=grad-torch.matmul(torch.matmul(M,M.permute(0,2,1)),grad)
        try:
            M = self.retraction(M,grad_R)
        except:
            print('svd')
        #grad_R=grad
        #M=M-self.lr*grad_R

        return M,state

