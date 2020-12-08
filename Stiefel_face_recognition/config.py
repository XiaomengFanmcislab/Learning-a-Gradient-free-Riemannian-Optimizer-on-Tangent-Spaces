import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--DIM', type=int, default=1024)
    parser.add_argument('--outputDIM', type=int, default=38)
    parser.add_argument('--batchsize_para', type=int, default=1)
    parser.add_argument('--batchsize_data', type=int, default=64)
    parser.add_argument('--datapath', type=str, default='data')
    parser.add_argument('--prepath', type=str, default='')
    parser.add_argument('--prepath2', type=str, default='')
    #parser.add_argument('--prepath', type=str, default='/home/mcislab/gaozhi/meta_metriclearning/meta_metriclearning_learnLSTMoptimizer_two/d128_twoLSTMtwo_updatestate_Optimizeelr_0.0005_0.01_mnist_meanvar_devide2_3000.pth')
    
    parser.add_argument('--nThreads', type=int, default=0)

    parser.add_argument('--Decay', type=int, default=40000)
    parser.add_argument('--modelsave', type=int,default=4000)
    parser.add_argument('--Decay_rate', type=float,default=0.8)

    parser.add_argument('--Pretrain', type=bool,default=False)
    parser.add_argument('--Imcrementflag', type=bool,default=False)
    parser.add_argument('--Imcrement', type=int,default=10000)
    parser.add_argument('--Observe', type=int, default=10)
    parser.add_argument('--Epochs', type=int, default=1000000)
    parser.add_argument('--Optimizee_Train_Steps', type=int, default=5000)
    parser.add_argument('--train_steps', type=int,default=50)

    parser.add_argument('--optimizer_lr', type=float, default=5e-5)
    parser.add_argument('--hand_optimizer_lr', type=float, default=0.000001)
    parser.add_argument('--Sample_number',type=int,default=60000)
    args = parser.parse_args()
    return args