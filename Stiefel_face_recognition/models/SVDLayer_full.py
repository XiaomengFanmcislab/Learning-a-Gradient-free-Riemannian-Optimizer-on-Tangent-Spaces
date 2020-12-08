import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function

'''
not correct
'''

class SVDLayerF(Function):
    @staticmethod
    def forward(self,input):

        n=(input.shape)[0]
        n1=(input.shape)[1]
        n2=(input.shape)[2]
        n_min=min(n1,n2)

        U=torch.zeros(n,n1,n1).cuda()
        S=torch.zeros(n,n1,n2).cuda()
        V=torch.zeros(n,n2,n2).cuda()

        for i in range(n):
            leftvector, value, rightvector=torch.svd( input[i],some=False)
            U[i]=leftvector
            S[i,0:n_min,:]=torch.diag(value)
            V[i]=rightvector

            
        self.save_for_backward(input, U,S,V)
        #print('EigLayer finish')
        return U,S,V


    @staticmethod
    def backward(self, grad_U,grad_S,grad_V ):

        input, U,S,V = self.saved_tensors

        n=input.shape[0]
        n1=input.shape[1]
        n2=input.shape[2]
        n_min=min(n1,n2)

        S_n=S[:,0:n_min,:]
        U1=U[:,:,0:n_min]
        U2=U[:,:,n_min:n1]
        grad_U1=grad_U[:,:,0:n_min]
        grad_U2=grad_U[:,:,n_min:n1]

        grad_input=Variable(torch.zeros(n,n1,n2 )) .cuda()

        e=torch.eye(n_min).cuda()
        P_i=torch.matmul(S_n,torch.ones(n_min,n_min).cuda())
        P_i=P_i*P_i
        P=(P_i-P_i.permute(0,2,1))+e
        epo=(torch.ones(P.shape).cuda())*0.000001
        P=torch.where(P!=0,P,epo)
        P=(1/P)-e

        o=torch.ones(n_min,n_min).cuda()
        S_n1=1/(S_n+o.repeat(n,1,1)-e.repeat(n,1,1))-o.repeat(n,1,1)+e.repeat(n,1,1)
        
        '''
        S_n1=torch.zeros(S_n.shape)
        for i in range(n):
            print(S_n[i])
            S_n1[i]=torch.inverse(S_n[i])
        '''
        D=torch.matmul(grad_U1,S_n1)-torch.matmul(torch.matmul(torch.matmul(U2,grad_U2.permute(0,2,1)),U1),S_n1)


        T1=torch.matmul(D,V.permute(0,2,1))

        d=torch.zeros(S.shape).cuda()
        d[:,0:n_min,:]=e.repeat(n,1,1)
        g2=grad_S-torch.matmul(U.permute(0,2,1),D)
        g2=g2*d
        T2=torch.matmul(torch.matmul(U,g2),V.permute(0,2,1))


        g3_VDUS=torch.matmul(V,D.permute(0,2,1))
        g3_VDUS=torch.matmul(g3_VDUS,U)
        g3_VDUS=torch.matmul(g3_VDUS,S)
        g3_VDUS=torch.matmul(V.permute(0,2,1),grad_V-g3_VDUS)
        g3_VDUS=P*g3_VDUS
        g3=(g3_VDUS+g3_VDUS.permute(0,2,1))/2
        T3=2*torch.matmul(torch.matmul(torch.matmul(U,S),g3),V.permute(0,2,1))


        grad_input=T1+T2+T3

        #print('grad_input',torch.sum(grad_input))

        return grad_input




class SVDLayer(nn.Module):
    def __init__(self):
        super(SVDLayer, self).__init__()
    

    def forward(self, input1):
        return SVDLayerF().apply(input1)

