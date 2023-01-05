import torch
from model import *
from util import *
import time
import os
class train_network():
    
    def __init__(self, train_dl,valid_dl,sim_test_dl,EPOCH,num_of_elements, ks, dropout, sig_length, sur_length,LR, wdecay, device,root_path):
        self.model = CNN1d(num_of_elements=num_of_elements,k_s=ks,\
        dp=dropout,sig_seq=sig_length,sur_seq=sur_length)
        self.device = device
        self.root_dir = root_path
        self.noe = num_of_elements
        self.model_dir = os.path.join(root_path,'savepoints',str(self.noe))
        self.test_dir = os.path.join(root_path,'savedata',str(self.noe))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=wdecay)
        self.loss = torch.nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor = 0.9,\
            verbose = True, patience = 5000, cooldown = 1,eps = 1e-8)
        self.EPOCH = EPOCH
        self.histtrain = []
        self.histValid = []
        self.histTest = []
        self.epochs = 0
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.sim_test_dl = sim_test_dl
    def val_cycle(self, test_dl):
        self.model.eval()
        with torch.no_grad():
            valid_loss = 0
            for x,y in test_dl:
                loss = self.loss(self.model(x),y)
                valid_loss += loss 
        return valid_loss,self.model(x),y
    def save(self):
        savepoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            'lr_scheduler': self.scheduler,
            'train_losses': self.histtrain,
            'valid_losses': self.histValid,
            'test_losses': self.histTest,
        }
        name = f"model_ep_{self.epochs}.pkl"
        torch.save(savepoint, os.path.join(self.model_dir, name))
        
    def save_test_output(self,testout,testlabel):
        # save the test output ( .mat file format )
        testout = testout.cpu().data.numpy()
        testlabel = testlabel.cpu().data.numpy()
        scio.savemat(os.path.join(self.test_dir,'test_output.mat'),{"testout": testout,"testlabel":testlabel})

    def load(self,epoch):
        savepoint = torch.load(os.path.join(self.model_dir, f"model_ep_{epoch}.pkl"))
        self.model.load_state_dict(savepoint['model'])
        self.optimizer.load_state_dict(savepoint['optimizer'])
        self.epochs = savepoint['epochs']
        self.lr_scheduler = savepoint['lr_scheduler']
        self.epochs = epoch
        self.train_losses = savepoint['train_losses']
        self.valid_losses = savepoint['valid_losses']
        self.test_losses = savepoint['test_losses']
    def train(self,every_print=100):
        time_start = time.time()
        print('Start training the network (number of sensors:' + str(self.noe) + ')')
        print('****************************************************************************')
        self.model.to(self.device)
        for epoch in range(self.EPOCH):
            self.epochs += 1
            #self.model.to(self.device)
            self.model.train()
            total_loss = 0
            for step, (b_x,b_y) in enumerate(self.train_dl):
                self.optimizer.zero_grad() 
                output = self.model(b_x)
                loss = self.loss(output,b_y)    

                loss.backward()  
                self.optimizer.step()
                #self.scheduler.step()
                loss = loss.item()
                total_loss += loss     
            
            total_loss = total_loss / (step+1)
            self.histtrain.append(total_loss)

            valid_loss,_,_ = self.val_cycle(self.valid_dl)
            self.histValid.append(valid_loss)

            test_loss,testout,testlabel = self.val_cycle(self.sim_test_dl)
            self.histTest.append(test_loss)   

            Use_time = (time.time()- time_start) / 60 ## minutes
            if (epoch+1) % every_print == 0:
                print(f"epoch {self.epochs}, "
                    f"train loss {total_loss:.4f}",
                    f"valid loss {valid_loss:.4f}",
                    f"test loss {test_loss:.4f}",
                    f"Total used time {Use_time:.1f} min")
                self.save()
        self.save_test_output(testout,testlabel)
        print('****************************************************************************')
        print('Training process finished')
        return self.model
            