import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function

from models.EigLayer import EigLayer
from models.m_sqrt import M_Sqrt
from models.m_exp import M_Exp

class Retraction(nn.Module):
    def __init__(self,lr):
        super(Retraction, self).__init__()

        self.beta=lr

        self.eiglayer1=EigLayer()
        self.eiglayer2=EigLayer()
        self.msqrt1=M_Sqrt(1)
        self.msqrt2=M_Sqrt(-1)
        self.mexp=M_Exp()


    def forward(self, inputs, grad):

        new_point=torch.zeros(inputs.shape).cuda()
        n=inputs.shape[0]

        P=-self.beta*grad
        PV=inputs+P

        for i in range(n):
            U,S,Y=torch.svd(PV[i])
            new_point[i]=torch.mm(U,Y.permute(0,1))

        return new_point
