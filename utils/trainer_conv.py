import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os


class Trainer():
    def __init__(self,model,train_set,valid_set,test_set,num_workers=4,Lr=0.001) -> None:
        self.device = torch.device("cuda:3")
        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_set)
        self.valid_loader = DataLoader(valid_set)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=Lr)
        self.records = {'val_losses': []}


    def train_iterations(self):
        self.model.train()
        losses = []
        for i,data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            in_data = data[0].to(torch.float32).to(self.device)
            # in_data = in_data.unsqueeze(1)
            output = self.model(in_data)
            ori_data = data[1].to(torch.float32).to(self.device)
            ori_data = ori_data.squeeze(1)
            loss = self.loss_fn(output,ori_data)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        trn_loss = np.array(losses).mean()
        return trn_loss


    def valid_iterations(self,mode='valid'):
        self.model.eval()
        if mode == 'test': loader = self.test_loader
        if mode == 'valid': loader = self.valid_loader
        losses = []
        with torch.no_grad():
            for i,data in enumerate(loader):
                in_data = data[0].to(torch.float32).to(self.device)
                # in_data = in_data.unsqueeze(1)
                output = self.model(in_data)
                ori_data = data[1].to(torch.float32).to(self.device)
                ori_data = ori_data.squeeze(1)
                loss = self.loss_fn(output,ori_data)
                losses.append(loss.cpu())
        val_loss = np.array(losses).mean()
        return val_loss



    def train(self,epochs=10):
        for epoch in range(epochs):
            train_loss = self.train_iterations()
            val_loss = self.valid_iterations()
            self.records['val_losses'].append(val_loss)
            save_log = ''
            if val_loss == np.array(self.records['val_losses']).min():
                save_log = 'save best model'
                self.save_model()
            print('epoch: {} train_loss: {} val_loss: {}, {}'.format(epoch, train_loss, val_loss, save_log))
            # self.test_model(epoch)

        self.load_ckpt('best_model.ckpt')
        test_loss = self.valid_iterations(mode='valid')
        print('test_loss: ', test_loss)


    def save_model(self):
        file_name = 'best_model.ckpt'
        with open(os.path.join(file_name),'wb') as f:
            torch.save({'model_state_dict': self.model.state_dict()}, f)

    def load_ckpt(self, ckpt_path='best_model.ckpt'):
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        

    def draw_pict(self):
        self.load_ckpt()
        self.model.eval()
        loader = self.valid_loader
        with torch.no_grad():
            for i,data in enumerate(loader):
                in_data = data[0].to(torch.float32).to(self.device)
                # in_data = in_data.unsqueeze(1)
                output = self.model(in_data)
                ori_data = data[1].to(torch.float32).to(self.device)
                ori_data = ori_data.squeeze(1)
                output_np = np.array(output.cpu())
                ori_data_np = np.array(ori_data.cpu())
                plt.scatter(ori_data_np,output_np,alpha=0.1,s=2)
                plt.savefig('aaa.png')




