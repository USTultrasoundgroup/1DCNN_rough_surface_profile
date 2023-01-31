import numpy as np
import torch
import scipy.io as scio
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os
import random

class StandardNormalized():
    """
    Normalized or inverse the input matrix
    """

    def __init__(self, maxx, minn):
        self.maxx = maxx
        self.minn = minn

    def transform(self, data):
        return (data - self.minn) / (self.maxx-self.minn)

    def inverse_transform(self, data):
        #Nmax = self.Nmax
        #Nmin = self.Nmin
        #print(self.minn)
        return data * (self.maxx - self.minn) + self.minn

class load_datasets():
    """
    Load the datasets (Train, test, exp)
    """
    def __init__(self,arraylist,loadpath,noe):
        self.arraylist = arraylist
        #self.init_input = init_input
        self.loadpath = loadpath
        self.noe = noe
    def loadeddata(self):
        l1 = scio.loadmat(self.loadpath)  #object
        init_input = torch.Tensor(l1['signallist'])  # The signal matrix has already been normalized to (0,1)
        target = torch.Tensor(l1['surfacelist'])
        input = selected_array_elements(init_input=init_input,arraylist=self.arraylist,num_of_element=self.noe)
        return input,target

def loadNormal(loadpath):
    l1 = scio.loadmat(loadpath)  #object
    surmax = torch.Tensor(l1['surmax'])  # The signal matrix has already been normalized to (0,1)
    surmin = torch.Tensor(l1['surmin'])
    return surmax,surmin
    #def forword(self) :
        #input,target = self.loadeddata()
        #return input,target

class DataLoader():
    def __init__(self, xs, ys,split_rate = 0.05, batch_size=512,random_seed=54,device='cuda:0', whether_test = 0):
        """
        :param xs:  
        :param ys:
        :param batch_size:
        :param split_rate: rate of choosing validation dataset from learning dataset
        """
        self.batch_size = batch_size
        self.bs = xs.shape[0]
        self.sb = int(split_rate*self.bs)
        #self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.rs = random_seed
        self.whether_test = whether_test
        self.random_seed = random_seed
        self.device = device
    def train_loader(self,xs,ys):
        train_dataset = Data.TensorDataset(xs.to(self.device), ys.to(self.device))
        train, test = Data.random_split(dataset= train_dataset, lengths=[self.bs-self.sb,self.sb]\
            ,generator = torch.Generator().manual_seed(self.random_seed))
        _train_loader = Data.DataLoader(dataset=train, batch_size= self.batch_size, shuffle=True)
        _valid_loader = Data.DataLoader(dataset= test, batch_size = self.batch_size,shuffle= False)
        return _train_loader,_valid_loader
    def test_loader(self,xs,ys):
        test_dataset = Data.TensorDataset(xs.to(self.device), ys.to(self.device))
        _test_loader = Data.DataLoader(dataset=test_dataset, batch_size= self.bs, shuffle=False)
        return _test_loader
    def forward(self):
        if self.whether_test == 0:
            return self.train_loader(self.xs,self.ys)
        else: 
            return self.test_loader(self.xs,self.ys)

def plot_and_save_test_results(exp_test_dl,model,title,savepath,noe,re_normalized=1):
    for x,y in exp_test_dl:
        total_num = x.shape[0]
        slice = np.arange(1,total_num+1)
        #print(slice)
        if total_num>10:  # randomly select 10 batches of dataset to plot
            slice = np.sort(np.random.choice(slice,10,replace=False))
            x = x[slice,:,:]
            y = y[slice,:]
        plot_num = x.shape[0]
        expexp = model(x)
        surmax,surmin = loadNormal(loadpath= r'./database/simulated_datasets.mat')
        nor  = StandardNormalized( surmax.squeeze(),surmin.squeeze())

        expexp = nor.inverse_transform(expexp)
        #print(expexp)
        if re_normalized == 1:
            y = nor.inverse_transform(y)
            #print(y.shape)
        plt.figure()
        for pp in range(plot_num):
            plt.subplot(2,int(plot_num/2),pp+1)
            plt.cla()
            plt.plot(expexp[pp].cpu().data.numpy(), label='exp')
            #plt.plot(expsim[pp].cpu().data.numpy(), label='sim')
            plt.plot(y[pp].cpu().data.numpy(), label='label')
            plt.ylim([-1,1])
            plt.legend()
            plt.title(title + ' index: '+str(slice[pp]))
        figname = title+'_'+str(noe)
        #plt.get_current_fig_manager().frame.Maximize(True)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized() # For QT backend only
        plt.show()
        plt.savefig(os.path.join(savepath,figname))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def selected_array_elements(init_input,arraylist,num_of_element):
    """
        :param init_input: The input signal matrix (Dimension: [B,C,S])
        B: Batch size, C: Channel number, S: Length of time sequence 
        :param arraylist: The horizontal cooridinate of all the array elements
        :param num_of_element: The number of used array elements for training neural network
    """
    if num_of_element == 32:
        input = init_input
    else:
        surfacenode = np.linspace(-8,8,num_of_element+1,endpoint=True)
        space = (surfacenode[-1]-surfacenode[-2])/2
        midnode = surfacenode[:num_of_element]+space

        indexlist = np.zeros(num_of_element)
        for ss in range(num_of_element):
            nearest = arraylist - midnode[ss]
            indexlist[ss] = np.argmin(np.abs(nearest))
        #print('The index of used transducers:',indexlist+1)
        input = init_input[:,indexlist.astype('int64'),:]

        
        #print(torch.max(input.flatten()))
        #expt = exptest[:,indexlist.astype('int64'),:]
       # expl = expsimu[:,indexlist.astype('int64'),:]
    return input
