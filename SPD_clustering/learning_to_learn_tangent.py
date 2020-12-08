import torch
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import time
import math, random

from ReplyBuffer import ReplayBuffer
from models.EigLayer import EigLayer
from models.EigLayer_loss import EigLayer_loss
from models.m_exp import M_Exp
from models.m_log import M_Log

from retraction import Retraction
import numpy as np

eiglayer1=EigLayer()
eiglayer_loss=EigLayer_loss()
mlog=M_Log()
mexp=M_Exp()
retraction=Retraction(1)


device=torch.device("cuda:0")

def f(inputs,M,label,sample_num):
   

    n=inputs.shape[0]
    loss=0
    for i in range(n):
        l=label[i]-1
        for j in range(sample_num):
            
            AIM=torch.mm(torch.mm(inputs[i],M[l*sample_num+j]),inputs[i])
            AIM=torch.unsqueeze(AIM,0)
            M_S,M_U=eiglayer_loss(AIM)
            M_Sl=mlog(M_S)
            AIM=torch.matmul(torch.matmul(M_U,M_Sl),M_U.permute(0,2,1))
            
            #AIM=inputs[i]-M[l*sample_num+j]
            #AIM=torch.unsqueeze(AIM,0)

            p=torch.sum(torch.sum(torch.pow(AIM,2),2),1)
            loss=loss+torch.mean(p)
    loss=loss/(n*sample_num)
    return loss


def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def Learning_to_learn_global_training(opt,hand_optimizee,optimizee,train_loader,train_loader_all):

    
    DIM=opt.DIM
    batchsize_para=opt.batchsize_para
    Observe=opt.Observe
    Epochs=opt.Epochs
    Optimizee_Train_Steps=opt.Optimizee_Train_Steps
    optimizer_lr=opt.optimizer_lr
    Decay=opt.Decay
    Decay_rate=opt.Decay_rate
    Imcrement=opt.Imcrement

    #adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)
    adam_global_optimizer = torch.optim.Adamax(optimizee.parameters(),lr = optimizer_lr)

    RB_list=[]
    for i in range(opt.category_num):
        RB=ReplayBuffer(opt.Content*opt.sample_num)  
        RB_list.append(RB)


    for i in range(Observe):

        if i ==0:
            M=torch.randn(batchsize_para,DIM, DIM).cuda()
            for k in range(batchsize_para):
                M[k]=torch.eye(DIM).cuda()

            state = (torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                     ) 
            iteration=torch.zeros(batchsize_para)
            M.requires_grad=True

            num=0
            for k in range(opt.category_num):
                for l in range(opt.sample_num):
                    RB_list[k].push((state[0][num],state[1][num],state[2][num],state[3][num]),M[num],iteration[num])
                    num=num+1

            count=1
            print ('observe finish',count)
            localtime = time.asctime( time.localtime(time.time()) )
            print ("local time", localtime)
            
        break_flag=False
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            
            loss = f(inputs,M,labels,opt.sample_num)
            loss.backward()

            M, state = hand_optimizee(M.grad, M, state)

            P_S,P_U=eiglayer_loss(M)
            P_Sl=mlog(P_S)

            iteration=iteration+1
            
            for k in range(batchsize_para):
                if iteration[k]>=Optimizee_Train_Steps-opt.train_steps:       
                    M[k]=torch.eye(DIM).cuda()
                    state[0][k]=torch.zeros(DIM,DIM).cuda()
                    state[1][k]=torch.zeros(DIM,DIM).cuda()
                    state[2][k]=torch.zeros(DIM,DIM).cuda()
                    state[3][k]=torch.zeros(DIM,DIM).cuda()   
                    iteration[k]=0
            

            state = (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach())
            M=M.detach()
            M.requires_grad=True

            num=0
            for k in range(opt.category_num):
                for l in range(opt.sample_num):
                    RB_list[k].push((state[0][num],state[1][num],state[2][num],state[3][num]),M[num],iteration[num])
                    num=num+1

            count=count+1
            print ('loss',loss)
            print ('observe finish',count)
            localtime = time.asctime( time.localtime(time.time()) )
            print ("local time", localtime)

            if count==Observe:
                break_flag=True
                break
        if break_flag==True:
            break                         

    for i in range(opt.category_num):
        RB_list[i].shuffle()



    check_point=optimizee.state_dict()
    check_point2=optimizee.state_dict()
    for i in range(Epochs): 
        print('\n=======> global training steps: {}'.format(i))
        if i % Decay==0 and i != 0:
            count=count+1
            adjust_learning_rate(adam_global_optimizer, Decay_rate)

        if opt.Imcrementflag==True:
            if i % Imcrement==0 and i != 0:
                Optimizee_Train_Steps=Optimizee_Train_Steps+70

        if i % opt.savemodel==0 and i != 0:
            if opt.loadpretrain == True:
                torch.save(optimizee.state_dict(), 'STATE/'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_Imcrement'+str(opt.Imcrement)+'d5_twoLSTM_Optimizee_lr0.001_iteration10000_nomoment_30_3_d5_twoLSTMtwo_updatestate_Optimizeelr_2_5000'+'.pth')
            if opt.loadpretrain == False:
                torch.save(optimizee.state_dict(), 'STATE/'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'_Imcrement'+str(opt.Imcrement)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+''+'.pth')

        if i==0:
            global_loss_graph=0
            train_loss=0
        else:
            global_loss_graph=global_loss_graph.detach()
            global_loss_graph=0
            train_loss=train_loss.detach()
            train_loss=0

        M_old=(torch.randn(batchsize_para,DIM, DIM)).cuda()
        state_old = (torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                torch.zeros(batchsize_para,DIM,DIM).cuda(),
                                ) 
        iteration_old=torch.zeros(batchsize_para)

        num=0
        for k in range(opt.category_num):
            state, M, iteration= RB_list[k].sample(opt.sample_num)
            for l in range(opt.sample_num):
                M_old[num]=M[l].detach()
                iteration_old[num]=iteration[l].detach()
                state_old[0][num]=state[l][0].detach()
                state_old[1][num]=state[l][1].detach()
                state_old[2][num]=state[l][2].detach()
                state_old[3][num]=state[l][3].detach()
                num=num+1
        M_old.requires_grad = True
        M_old.retain_grad()
        

        break_flag=False
        flag=False
        count=0
        adam_global_optimizer.zero_grad()
        while(1):
            for j, data in enumerate(train_loader, 0):

                # print('---------------------------------------------------------------------------')
                #print('M',M)
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels).cuda()
                inputs_shape=inputs.shape
                data_set=inputs_shape[0]

                if data_set!=opt.batchsize_data:
                    break

               

                if count==0:

                    loss_old = f(inputs,M_old,labels,opt.sample_num)
                    train_loss=train_loss+loss_old
                    loss_old.backward(retain_graph=True)
                    M_grad=M_old.grad.data
                    print('M_grad',(torch.sum(M_grad)/(M_grad.shape[0])))
                    P=torch.matmul(torch.matmul(M_old, (M_grad+M_grad.permute(0,2,1))/2),M_old)
                    P=P*5
                    
                    # P=(M_grad+M_grad.permute(0,2,1))/2
                    print('P',(torch.sum(P)/(P.shape[0])))
                    flag,M_csgd=retraction(M_old,P,1)

                    for j_all, data_all in enumerate(train_loader_all, 0):
                        inputs_all, labels_all = data_all
                        inputs_all=inputs_all.to(device)
                        labels_all=labels_all.to(device)
                        loss_csgd=f(inputs_all,M_csgd,labels_all,opt.sample_num)
                        break
                    print('EPOCHES:{},loss_csgd:{}'.format(i,loss_csgd.item()))
                


                lr, update, state_old = optimizee(P, state_old, inputs)

                # lr, update, state_old = optimizee(M_old.grad, state_old)
                #update=update+M_old.grad
                s=torch.sum(state_old[0])+torch.sum(state_old[1])+torch.sum(state_old[2])+torch.sum(state_old[3])

                if s > 100000:
                    break_flag=True
                    flag=True
                    break


                update_R=(update+update.permute(0,2,1))/2
                # update_R=torch.matmul(torch.matmul(M_old, (update+update.permute(0,2,1))/2),M_old)

                # print('lr',(torch.sum(lr)/(lr.shape[0])))
                # print('update',(torch.sum(update)/(update.shape[0])))
                # print('update_R',(torch.sum(update_R)/(update_R.shape[0])))

                # P=P-1e-1*lr*update_R    
                P=P-lr/lr*update_R                       

                update.retain_grad()
                P.retain_grad()
                update_R.retain_grad()
                lr.retain_grad()

                count=count+1
                if count==opt.train_steps:
                    break_flag=True
                    break   

                if flag == True:
                    break_flag=True
                    break

            if break_flag==True:
                break 

        #adam_global_optimizer.zero_grad()
        print('M_old',(torch.sum(M_old)/(M_old.shape[0])))
        print('P',(torch.sum(P)/(P.shape[0])))
        flag,M_old = retraction(M_old,P,1)
        M_old.retain_grad()

        for j_all, data_all in enumerate(train_loader_all, 0):
            inputs_all, labels_all = data_all
            inputs_all=inputs_all.to(device)
            labels_all=labels_all.to(device)
            global_loss_graph=f(inputs_all,M_old,labels_all,opt.sample_num)
            break

        print('=======>global_loss_graph',global_loss_graph.item())
        global_loss_graph.backward()

        P_grad_shape=P.grad.shape
        #print('P_shape',P_grad_shape)
        number_P=1
        for number in P_grad_shape:
            number_P=number_P*number
        #print(number_P)
        P_grad_data=P.grad.data
        P_grad_mean=torch.sum(torch.norm(P_grad_data,p=1,dim=(0,1))).detach().cpu().numpy().tolist()
        P_grad_mean=P_grad_mean/number_P
        print('P_gradient',P_grad_mean)
        if np.isnan(P_grad_mean):
            print('ERROR NAN!!!')
            continue

        params = list(optimizee.named_parameters())
        (name,network_weight)=params[37]
        #print('network_weight',network_weight)
        network_weight_copy=network_weight.clone()

        
        #print(name)
        network_weight_grad_copy=network_weight.grad.data
        network_weight_shape=network_weight_grad_copy.shape
        network_weight_length=len(network_weight_shape)
        network_weight_size=1
        for l in range(network_weight_length):
            network_weight_size=network_weight_size*network_weight_shape[l]

        grad_mean=torch.sum(torch.norm(network_weight_grad_copy,p=1,dim=(0))).detach().cpu().numpy().tolist()
        grad_mean=grad_mean/network_weight_size
        print('network_grad_mean',grad_mean)
        if np.isnan(grad_mean):
            print('ERROR NAN!!!')
            continue


        iteration_old=iteration_old+1
        if flag==False:           
            adam_global_optimizer.step()

            params = list(optimizee.named_parameters()) 

            (name,network_weight_after)=params[37]
            contrast=network_weight_after- network_weight_copy
            #print(contrast)
            loss_con=torch.sum(torch.norm(contrast,p=1,dim=(0))).detach().cpu().numpy().tolist()
            loss_con=loss_con/network_weight_size
            print('loss_con',loss_con)
            length=len(params)
            for t in range(length):
                (name,param)=params[t]
                param.grad.data.zero_()

            for k in range(batchsize_para):
                if iteration_old[k]>=Optimizee_Train_Steps-opt.train_steps:
                    M_old[k]=torch.eye(DIM)
                    state_old[0][k]=torch.zeros(DIM,DIM).cuda()
                    state_old[1][k]=torch.zeros(DIM,DIM).cuda()
                    state_old[2][k]=torch.zeros(DIM,DIM).cuda()
                    state_old[3][k]=torch.zeros(DIM,DIM).cuda()     
                    iteration_old[k]=0
            
            num=0
            for k in range(opt.category_num):
                for l in range(opt.sample_num):
                    RB_list[k].push((state_old[0][num],state_old[1][num],state_old[2][num],state_old[3][num]),M_old[num],iteration_old[num])
                    num=num+1

            check_point=check_point2
            check_point2=optimizee.state_dict()
        else:
            print('=====>eigenvalue break, reloading check_point')
            optimizee.load_state_dict(check_point)

        print('=======>global_loss_graph',global_loss_graph.item())
