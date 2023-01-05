import torch
import math
class cal_d():
    def __init__(self,sq,k_s=11,mpk=2):
        self.k_s = k_s
        self.mpk = mpk
        self.sq = sq
    def outshape(self,):
        data = self.sq+2*0-(self.k_s-1)-1+1
        data = (data+2*0-(self.mpk-1)-1)/self.mpk+1
        return data


class CNN1d(torch.nn.Module):
    def __init__(self, num_of_elements, k_s=11, dp=0.1, mpk=2, sig_seq = 250,sur_seq=200):
        super(CNN1d,self).__init__()
        self.k_n1 = num_of_elements*2
        self.k_n2 = num_of_elements*4
        self.k_n3 = num_of_elements*8
        self.k_s = k_s
        self.mpk = mpk
        #self.bn0 = torch.nn.BatchNorm1d(num_of_elements,momentum=0.5)
        self.bn1 = torch.nn.BatchNorm1d(self.k_n1, momentum=0.5)
        self.bn2 = torch.nn.BatchNorm1d(self.k_n2, momentum=0.5)
        self.bn3 = torch.nn.BatchNorm1d(self.k_n3, momentum=0.5)
        self.shapeout1 = math.floor(cal_d(sq=sig_seq).outshape())
        #print(self.shapeout1)
        self.shapeout2 = math.floor(cal_d(sq=self.shapeout1).outshape())
        #print(self.shapeout2)
        self.shapeout3 = math.floor(cal_d(sq=self.shapeout2).outshape())*self.k_n3
        #self.dp = torch.nn.Dropout(0.1)     
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels= num_of_elements,
                out_channels = self.k_n1, 
                kernel_size = self.k_s,
                stride  = 1,
                padding = 0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size = self.mpk),
            torch.nn.Dropout(dp),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels= self.k_n1,
                out_channels = self.k_n2,
                kernel_size = self.k_s,
                stride  = 1,
                padding = 0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(self.mpk),
            torch.nn.Dropout(dp),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels= self.k_n2,
                out_channels = self.k_n3,
                kernel_size = self.k_s,
                stride  = 1,
                padding = 0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(self.mpk),
            torch.nn.Dropout(dp),
        )
        self.out = torch.nn.Sequential(

            torch.nn.Linear(self.shapeout3,sur_seq))

    def forward(self,x):
        x = self.conv1(x)
    
        x = self.bn1(x)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.conv3(x)

        x = self.bn3(x)

        
        x = x.view(x.size(0), -1)  
        output = self.out(x)
        return output