import numpy as np
import torch
import torch.nn.functional as F
#import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
#import xlrd
import math
import argparse
import scipy.io as scio
#import time
from datetime import datetime  
from util import *

from train import *
#from model import *
today = (datetime.now()).strftime('%m%d')  

parser = argparse.ArgumentParser()
parser.add_argument('--sim_Train_path',type=str,default=r'./database/simulated_datasets.mat',\
    help='simulated training datasets path')
parser.add_argument('--sim_Test_path',type=str,default=r'./database/simulated_testing_datasets.mat',\
    help='simulated testing datasets path')    
parser.add_argument('--exp_path',type=str,default=r'./database/experiment_datasets.mat',\
    help='Experiment datasets path')
parser.add_argument('--device',type=str,default='cuda:1',help='device for training')
parser.add_argument('--sig_length',type=int,default=250,help='the sequence length of signal')
parser.add_argument('--sur_length',type=int,default=200,help='the sequence length of surface')
parser.add_argument('--num_of_elements',type=int,default=32,help='number of array elements')
parser.add_argument('--random_seed',type=int,default=54,help='random seed')
parser.add_argument('--val_rate',type=float,default=0.05,help='rate of validation')
parser.add_argument('--batch_size',type=int,default=512,help='batch size')
parser.add_argument('--seed',type=int,default=5,help='random seed')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='learning rate')
parser.add_argument('--ks',type=float,default=11,help='kernel_size')
parser.add_argument('--dropout',type=float,default=0.1,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.00,help='weight decay rate')
parser.add_argument('--EPOCH',type=int,default=1500,help='total epoch')
parser.add_argument('--print_every',type=int,default=100,help='Print the loss every x epochs')
parser.add_argument('--savefig',type=str,default='./savemodel/figure',help='save path:figure')
parser.add_argument('--savepath',type=str,default='./savemodel',help='save path:profile')
parser.add_argument('--exp',type=int,default=1,help='whether display experiment results during training')
parser.add_argument('--nArray',type=int,default=32,help='number of array sensors')
parser.add_argument('--ArrayL',type=float,default=15.5,help='Length of array probe')

args = parser.parse_args()


def main():
    #LD = load_datasets()

    arraylist = np.linspace(-args.ArrayL/2,args.ArrayL/2,args.nArray)  # Calculate the coordinate of array sensors
    STI = load_datasets(arraylist,args.sim_Train_path,args.num_of_elements)
    sim_train_input,sim_train_target = STI.loadeddata()

    STEI = load_datasets(arraylist,args.sim_Test_path,args.num_of_elements)
    sim_test_input,sim_test_target = STEI.loadeddata()

    EI = load_datasets(arraylist,args.exp_path,args.num_of_elements)
    exp_input,exp_target = EI.loadeddata()

    train_dl,valid_dl = DataLoader(sim_train_input,sim_train_target,split_rate=args.val_rate,batch_size=args.batch_size,\
    random_seed=args.random_seed,whether_test=0,device = args.device).forward()

    sim_test_dl = DataLoader(sim_test_input,sim_test_target,whether_test=1,device = args.device).forward()
    exp_test_dl = DataLoader(exp_input,exp_target,whether_test=1,device = args.device).forward()
    
    Trained_model = train_network(train_dl, valid_dl, sim_test_dl, EPOCH=args.EPOCH, num_of_elements=args.num_of_elements,ks=args.ks,dropout=args.dropout,sig_length=args.sig_length,\
 sur_length=args.sur_length, LR = args.learning_rate, wdecay=args.weight_decay, device=args.device, root_path=args.savepath).train()
    print('parameters_count:',count_parameters(Trained_model))

  # train_network()
    plot_and_save_test_results(exp_test_dl,Trained_model,'Experiment results',args.savefig,noe = args.num_of_elements,re_normalized=0)
    #sim_test_dl_s = Data.random_split()
    plot_and_save_test_results(sim_test_dl,Trained_model,'Simulated testing results',args.savefig,noe = args.num_of_elements,re_normalized=1)
    #plot_test()

main()