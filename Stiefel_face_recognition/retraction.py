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
        for i in range(n):
            temp=PV[i]
            temp_mul=torch.transpose(temp,0,1).mm(temp)

            e_0,v_0=torch.symeig(temp_mul,eigenvectors=True)
            e_0=abs(e_0)+1e-6
            e_0=e_0.pow(-0.5)
            temp1=v_0.mm(torch.diag(e_0)).mm(torch.transpose(v_0,0,1))

            temp=temp.mm(temp1)
            new_point[i]=temp


        return new_point