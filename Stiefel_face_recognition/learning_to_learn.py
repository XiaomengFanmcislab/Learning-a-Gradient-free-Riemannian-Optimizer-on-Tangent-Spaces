import torch
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import time
import math, random
import numpy as np 

from losses.LOSS import ContrastiveLoss
from ReplyBuffer import ReplayBuffer
from retraction import Retraction

retraction=Retraction(1)
grad_list = []

def print_grad(grad):
    grad_list.append(grad)

def nll_loss(data,label):
    n=data.shape[0]
    L=0
    for i in range(n):
        L=L+torch.abs(data[i][label[i]])
    return L


def f(inputs,target,M):
    
    loss_function=torch.nn.CrossEntropyLoss(reduction='sum')
    #M=M.permute(0,2,1)
    #print('inputs:{}'.format(inputs.shape))
    #print('M:{}'.format(M.shape))
    X=torch.matmul(inputs,M)
    n=M.shape[0]
    L=0
    for i in range(n):
        X_temp=torch.squeeze(X[i],0)
        #y_pred=torch.nn.functional.log_softmax(X_temp)
        #print(y_pred)
        target_temp=torch.squeeze(target[i],0)
        #print(X_temp)
        #print(target_temp)
        #L=L+nll_loss(y_pred,target_temp)
        L=L+loss_function(X_temp,target_temp)

    return L



def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def learning_rate_multi(lr,update):
    n=lr.shape[0]
    for i in range(n):
        learning_rate=lr[i]
        update_temp=update[i]*learning_rate
        update[i]=update_temp
        #print('---------temp-----------',update_temp.shape)
    #print(update)
    #print('---------------',update.shape)
    return update

def Learning_to_learn_global_training(opt,hand_optimizee,optimizee,train_loader):

    DIM=opt.DIM
    outputDIM=opt.outputDIM
    batchsize_para=opt.batchsize_para
    Observe=opt.Observe
    Epochs=opt.Epochs
    Optimizee_Train_Steps=opt.Optimizee_Train_Steps
    optimizer_lr=opt.optimizer_lr
    Decay=opt.Decay
    Decay_rate=opt.Decay_rate
    Imcrement=opt.Imcrement
    Sample_number=opt.Sample_number
    X=[]
    Y=[]
    network_contrast=[]
    network_gradient=[]

    M_after_gradient=[]
    P_gradient=[]
    update_gradient=[]
    M_update_gradient=[]

    Number_iterations=0


    data1=np.load('data/YaleB_train_3232.npy')
    data2=torch.from_numpy(data1)
    data2=data2.float()
    data3=data2.to(torch.device("cuda:0"))

    



    LABELS=np.load('data/YaleB_train_gnd.npy')
    LABELS=torch.from_numpy(LABELS)
    LABELS=LABELS.long()
    LABELS=LABELS-1
    #LABELS=LABELS.squeeze()
    LABELS=LABELS.to(torch.device("cuda:0"))
    LABELS=LABELS.squeeze()
    
    data_all=data3.view(batchsize_para,data3.shape[0]//batchsize_para,-1)
    LABELS=LABELS.view(batchsize_para,LABELS.shape[0]//batchsize_para)


    #adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)
    adam_global_optimizer = torch.optim.Adamax(optimizee.parameters(),lr = optimizer_lr)
    f_loss_optimizer=nn.MSELoss()

    RB=ReplayBuffer(500*batchsize_para)

    Square=torch.eye(DIM)

    for i in range(Observe):
        RB.shuffle()
        if i ==0:
            M=torch.randn(batchsize_para,DIM, outputDIM).cuda()
            for k in range(batchsize_para):
                nn.init.orthogonal_(M[k])

            state = (torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                     torch.zeros(batchsize_para,DIM,outputDIM).cuda(),
                                     ) 
            iteration=torch.zeros(batchsize_para)
            #M.retain_grad()
            M.requires_grad=True

            RB.push(state,M,iteration)  
            count=1
            print ('observe finish',count)

        break_flag=False
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
            labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para)
            #inputs=inputs.permute(0,2,1)
            loss = f(inputs,labels,M)
            
            loss.backward()
            #M_grad=M.grad
            M, state = hand_optimizee(M.grad, M, state)

            print('-------------------------')
            #print('MtM', torch.mm(M[k].t(),M[k]))

            iteration=iteration+1
            for k in range(batchsize_para):
                if iteration[k]>=Optimizee_Train_Steps-opt.train_steps:
                    M[k]=Square[:,0:outputDIM]
                    state[0][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[1][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[2][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[3][k]=torch.zeros(DIM,outputDIM).cuda()   
                    iteration[k]=0


            state = (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach())
            M=M.detach()
            M.requires_grad=True
            M.retain_grad()
            
            RB.push(state,M,iteration)
            count=count+1
            print ('loss',loss)
            print ('observe finish',count)
            localtime = time.asctime( time.localtime(time.time()) )

            if count>=Observe:
                break_flag=True
                break
        if break_flag==True:
            break                         
    
    RB.shuffle()

    check_point=optimizee.state_dict()
    check_point2=optimizee.state_dict()
    check_point3=optimizee.state_dict()
    Global_Epochs=0
    for i in range(Epochs): 
        print('\n=======> global training steps: {}'.format(i))
        if (i+1) % Decay==0 and (i+1) != 0:
            count=count+1
            adjust_learning_rate(adam_global_optimizer, Decay_rate)

        if opt.Imcrementflag==True:
            if (i+1) % Imcrement==0 and (i+1) != 0:
                Optimizee_Train_Steps=Optimizee_Train_Steps+50

        if (i+1) % opt.modelsave==0 and (i+1) != 0:
            print('-------------------------------SAVE----------------------------------------------')
            print(opt.modelsave)
            if opt.Pretrain==True:
                torch.save(optimizee.state_dict(), 'STATE/'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'.pth')
            else:
                torch.save(optimizee.state_dict(), 'STATE/'+str(i)+'_'+str(opt.optimizer_lr*1000)+'_Decay'+str(opt.Decay)+'_Observe'+str(opt.Observe)+'_Epochs'+str(opt.Epochs)+'_Optimizee_Train_Steps'+str(opt.Optimizee_Train_Steps)+'_train_steps'+str(opt.train_steps)+'_hand_optimizer_lr'+str(opt.hand_optimizer_lr)+'nopretrain_newlr_meanvar_devide2'+'.pth')
                
                '''
                M_after_gradient_numpy=np.array(M_after_gradient)
                P_gradient_numpy=np.array(P_gradient)
                update_gradient_numpy=np.array(update_gradient)
                M_update_gradient_numpy=np.array(M_update_gradient)
                network_gradient_numpy=np.array(network_gradient)
                network_contrast_numpy=np.array(network_contrast)
                np.save('/home/mcislab/Tangent_Space/stiefel/meta_metriclearning_pca_ouroptimizer_new/gradient/M_gradient.npy',M_after_gradient_numpy)
                np.save('/home/mcislab/Tangent_Space/stiefel/meta_metriclearning_pca_ouroptimizer_new/gradient/P_gradient.npy',P_gradient_numpy)
                np.save('/home/mcislab/Tangent_Space/stiefel/meta_metriclearning_pca_ouroptimizer_new/gradient/update_gradient.npy',update_gradient_numpy)
                np.save('/home/mcislab/Tangent_Space/stiefel/meta_metriclearning_pca_ouroptimizer_new/gradient/M_update_gradient.npy',M_update_gradient_numpy)
                np.save('/home/mcislab/Tangent_Space/stiefel/meta_metriclearning_pca_ouroptimizer_new/gradient/network_gradient.npy',network_gradient_numpy)
                np.save('/home/mcislab/Tangent_Space/stiefel/meta_metriclearning_pca_ouroptimizer_new/gradient/network_contrast.npy',network_contrast_numpy)              
                '''       

        if i==0:
            global_loss_graph=0
        else:
            global_loss_graph=global_loss_graph.detach()
            global_loss_graph=0

        state_read, M_read, iteration_read= RB.sample(batchsize_para) 
        state=(state_read[0].detach(),state_read[1].detach(),state_read[2].detach(),state_read[3].detach())
        M=M_read.detach()
        iteration=iteration_read.detach()
        M.requires_grad=True
        M.retain_grad()
        

        flag=False
        break_flag=False
        count=0
        new_count=0
        begin=True
        adam_global_optimizer.zero_grad()
        while(1):
            for j, data in enumerate(train_loader, 0):
                #print('---------------------------------------------------------------------------')
                #print('M',M)
                inputs, labels = data
                #inputs = Variable(inputs.cuda())
                #labels = Variable(labels).cuda()
                inputs=inputs.cuda()
                labels=labels.cuda()
                inputs=inputs.view(batchsize_para,inputs.shape[0]//batchsize_para,-1)
                labels=labels.view(batchsize_para,labels.shape[0]//batchsize_para)

                #inputs=inputs.permute(0,2,1)

                if count==0:
                    loss = f(inputs,labels,M)
                     
                    loss.backward(retain_graph=True)

                    
                    M_grad=M.grad.data
                    #print('M_grad',M_grad)
                    
                    temp=torch.matmul(M.permute(0,2,1),M_grad)
                    temp=0.5*(temp+temp.permute(0,2,1))


                    P=M_grad-torch.matmul(M,temp)
                    P=P*1e-6
                    P_data=P.clone()
                    

                    M_csgd=retraction(M,P,1)
                    loss_csgd=f(data_all,LABELS,M_csgd)
                    print('EPOCHES:{},loss_csgd:{}'.format(i,loss_csgd.item()/1900))
                    #P=M_grad-torch.matmul(torch.matmul(M,M.permute(0,2,1)),M_grad)
                    print('Epochs',i,'LOSS',loss.item()/opt.batchsize_data)

               
                lr, update, state = optimizee(P, state, inputs,labels)
                lr=torch.abs(lr)
               
                s=torch.sum(state[0])+torch.sum(state[1])+torch.sum(state[2])+torch.sum(state[3])
                if s > 100000:
                    break_flag=True
                    flag=True
                    break

                
                #projection
                M_temp=torch.matmul(M.permute(0,2,1),update)
                M_temp=0.5*(M_temp+M_temp.permute(0,2,1))
                M_update=update-torch.matmul(M,M_temp)
               
                #P=P-1e6*learning_rate_multi(lr,M_update)
                

                P=P-lr*M_update
               
                #print('lr',lr)
                #print('M_update',M_update)
                #print('P',P)
                #print('P_data',P_data)

                update.retain_grad()
                P.retain_grad()
                M_update.retain_grad()
                lr.retain_grad()
             
                count=count+1

                if count==opt.train_steps:
                    break_flag=True
                    break
            if break_flag==True:
                break

        iteration=iteration+1
        M = retraction(M,P,1)
        M.retain_grad()
        #print('M_after',M)
        #print('P',P)
        
        global_loss_graph=f(data_all,LABELS,M)
        
        global_loss_graph.backward(retain_graph=True)

        
        M_after_shape=M.grad.shape
        number_M=1
        for number in M_after_shape:
            number_M=number_M*number
        M_grad_after=M.grad.data
        M_grad_mean=torch.sum(torch.norm(M_grad_after,p=1,dim=(0,1))).detach().cpu().numpy().tolist()
        M_grad_mean=M_grad_mean/number_M
        print('M_after',M_grad_mean)

        if np.isnan(M_grad_mean):
            print('ERROR NAN!!!')
            continue

        M_after_gradient.append(M_grad_mean)
        


        update_grad_shape=update.grad.shape
        number_update=1
        for number in update_grad_shape:
            number_update=number_update*number
        update_grad_data=update.grad.data
        update_grad_mean=torch.sum(torch.norm(update_grad_data,p=1,dim=(0,1))).detach().cpu().numpy().tolist()
        update_grad_mean=update_grad_mean/number_update
        print('update_gradient',update_grad_mean)
        if np.isnan(update_grad_mean):
            print('ERROR NAN!!!')
            continue

        update_gradient.append(update_grad_mean)
        



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

        P_gradient.append(P_grad_mean)
        
        P.grad.data.zero_()


        M_update_shape=M_update.shape
        number_M_update=1
        for number in M_update_shape:
            number_M_update=number_M_update*number
        M_update_data=M_update.grad.data
        M_update_mean=torch.sum(torch.norm(M_update_data,p=1,dim=(0,1))).detach().cpu().numpy().tolist()
        M_update_mean=M_update_mean/number_M_update
        
        #print('M_update_gradient',M_update_gradient)
        print('M_update_mean',M_update_mean)
        if np.isnan(M_update_mean):
            print('ERROR NAN!!!')
            continue
        M_update_gradient.append(M_update_mean)

        
       
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

        #print('network_weight_shape',network_weight_shape)
        # print('network_weight_shape',network_weight_size)
        
        grad_mean=torch.sum(torch.norm(network_weight_grad_copy,p=1,dim=(0))).detach().cpu().numpy().tolist()
        grad_mean=grad_mean/network_weight_size
        print('network_grad_mean',grad_mean)
        if np.isnan(grad_mean):
            print('ERROR NAN!!!')
            continue
        network_gradient.append(grad_mean)
   

        if flag==False:
            adam_global_optimizer.step()
            
            params = list(optimizee.named_parameters()) 

            (name,network_weight_after)=params[37]
            contrast=network_weight_after- network_weight_copy
            #print(contrast)
            loss_con=torch.sum(torch.norm(contrast,p=1,dim=(0))).detach().cpu().numpy().tolist()
            loss_con=loss_con/network_weight_size
            network_contrast.append(loss_con)
            #print('network_weight_copy',network_weight_copy)
            #print('network_weight_after',network_weight_after)
            
            
            length=len(params)
            for t in range(length):
                (name,param)=params[t]
                param.grad.data.zero_()
           
            #print('network_weight_after',network_weight_after)
            print('EPOCHES:{},Parameters_update:{},loss_contrast:{}'.format(i,flag,loss_con))
            
            for k in range(batchsize_para):
                if iteration[k]>=Optimizee_Train_Steps-opt.train_steps:
                    nn.init.orthogonal_(M[k])
                    state[0][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[1][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[2][k]=torch.zeros(DIM,outputDIM).cuda()
                    state[3][k]=torch.zeros(DIM,outputDIM).cuda()     
                    iteration[k]=0
                    
            RB.push((state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach()),M.detach(),iteration.detach())            

            check_point=check_point2
            check_point2=check_point3
            check_point3=optimizee.state_dict()         
        else:
            print('=====>eigenvalue break, reloading check_point')
            optimizee.load_state_dict(check_point)

        print('==========>EPOCHES<-=========',i)
        print('=======>global_loss_graph',global_loss_graph.item()/1900)
