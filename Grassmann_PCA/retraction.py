import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function
from models.SVDLayer_full import SVDLayer

svd=SVDLayer()

class Retraction(nn.Module):
    def __init__(self,lr):
        super(Retraction, self).__init__()

        self.beta=lr


    def forward(self, inputs, grad,lr):


        new_point=torch.zeros(inputs.shape).cuda()
        n=inputs.shape[0]

        P=-lr*grad
        PV=inputs+P

        n1=(PV.shape)[1]
        n2=(PV.shape)[2]
        n_min=min(n1,n2)
        
        U,S,Y=svd(PV)
        new_point=torch.matmul(U[:,:,0:n_min],Y.permute(0,2,1))

        return new_point