import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import scipy.stats
import torch.nn as nn
from torch import Tensor

def spearmanr(x, y):
    '''
    x,y [bsz,,]
    '''
    x_rank = x.argsort(dim=2).float()
    y_rank = y.argsort(dim=2).float()
    # n = x.numel()
    # n = 10
    n = x.shape[-1]
    xy_rank = x_rank - y_rank
    return 1 - 6 * ((x_rank - y_rank) ** 2).sum(dim=2) / (n * (n ** 2 - 1))

# def spearman_batch(x,y):
#     bsz = x.shape[0]
#     spearman_lis = []
#     for i in range(bsz):
#         spearman_lis.append(spearmanr(x[i],y[i]))
#     return spearman_lis

class Trainer():
    def __init__(self,model,train_set,valid_set,test_set,batchsize=64,num_workers=4,Lr=0.01) -> None:
        # self.device = torch.device("cuda:3")
        # self.model = model.to(self.device)
        # f = open('/home/yanggk/Data/HW/attn.log','a')
        # self.writer = f
        self.model = torch.nn.DataParallel(model).cuda()
        self.batchsz = batchsize
        self.train_loader = DataLoader(train_set,batch_size=batchsize,num_workers=num_workers)
        self.valid_loader = DataLoader(valid_set)
        self.test_loader = DataLoader(test_set)
        # self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.loss_fn = torch.nn.L1Loss(reduction='mean')
        # self.loos_fn = torch.nn.CosineEmbeddingLoss()
        # self.loss_fn = torch.nn.Spearman_loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=Lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=50,min_lr=1e-06,verbose=True)
        self.records = {'val_losses': []}

    def compute_rank_correlation(self,x, y):
        """
        x,y :[bsz,1,length]
        """
        batch_sz = x.shape[0]
        x_mean = x.mean(2).unsqueeze(1)
        y_mean = y.mean(2).unsqueeze(1)
        d_x = x - x_mean
        d_y = y - y_mean
        upper = torch.sum(d_x * d_y,dim=2)
        downer = torch.sum(torch.square(d_x) * torch.square(d_y),dim=2)
        spear = upper/downer
        spearman = spear.mean()
        return torch.tensor(1) - spearman
        
    def train_iterations(self,mode='train'):
        if mode =='transport':
            self.load_ckpt()
        self.model.train()
        losses = []
        for i,data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # in_data = data[0].to(torch.float32).to(self.device)
            in_data = data[0].to(torch.float32).cuda()
            in_data = in_data.unsqueeze(1)
            output = self.model(in_data)
            # ori_data = data[1].to(torch.float32).to(self.device)
            ori_data = data[1].to(torch.float32).cuda()
            ori_data = ori_data.unsqueeze(1)
            loss = self.loss_fn(output,ori_data)
            # loss = self.compute_rank_correlation(output,ori_data)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.cpu().detach())
        # self.scheduler.step(loss)
        trn_loss = np.array(losses).mean()
        return trn_loss


    def valid_iterations(self,mode='valid'):
        self.model.eval()
        if mode == 'test': loader = self.test_loader
        if mode == 'valid': loader = self.valid_loader
        losses = []
        with torch.no_grad():
            for i,data in enumerate(loader):
                # in_data = data[0].to(torch.float32).to(self.device)
                in_data = data[0].to(torch.float32).cuda()
                in_data = in_data.unsqueeze(1)
                output = self.model(in_data)
                # ori_data = data[1].to(torch.float32).to(self.device)
                ori_data = data[1].to(torch.float32).cuda()
                ori_data = ori_data.unsqueeze(1)
                loss = self.loss_fn(output,ori_data)
                # loss = self.compute_rank_correlation(output,ori_data)
                losses.append(loss.cpu().detach())
        val_loss = np.array(losses).mean()
        return val_loss



    def train(self,epochs=500,mode='train'):
        for epoch in range(epochs):
            train_loss = self.train_iterations(mode)
            val_loss = self.valid_iterations()
            self.records['val_losses'].append(val_loss)
            self.scheduler.step(val_loss)
            save_log = ''
            if val_loss == np.array(self.records['val_losses']).min():
                save_log = 'save best model'
                self.save_model()
            print('epoch: {} train_loss: {} val_loss: {}, {}'.format(epoch, train_loss, val_loss, save_log))
            # self.test_model(epoch)
            if epoch % 100 == 0:
                self.draw_pict(epoch)

        self.load_ckpt('best_model.ckpt')
        test_loss = self.valid_iterations(mode='test')
        print('test_loss: ', test_loss)


    def save_model(self):
        file_name = 'best_model.ckpt'
        with open(os.path.join(file_name),'wb') as f:
            torch.save({'model_state_dict': self.model.state_dict()}, f)

    def load_ckpt(self, ckpt_path='best_model.ckpt'):
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        

    def draw_pict(self,epoch=100):
        self.load_ckpt()
        self.model.eval()
        loader = self.test_loader
        with torch.no_grad():
            ori_lis = []
            pred_lis = []
            for i,data in enumerate(loader):
                # in_data = data[0].to(torch.float32).to(self.device)
                in_data = data[0].to(torch.float32).cuda()
                in_data = in_data.unsqueeze(1)
                # in_data = in_data.permute(0,2,1)
                output= self.model(in_data)
                # ori_data = data[1].to(torch.float32).to(self.device)
                ori_data = data[1].to(torch.float32).cuda()
                ori_data = ori_data.unsqueeze(1)
                # ori_data.permute(0,2,1)
                output_np = np.array(output.cpu())
                ori_data_np = np.array(ori_data.cpu())
                ori_lis.append(ori_data_np)
                pred_lis.append(output_np)
            ori_np = np.array(ori_lis).flatten()
            pred_np = np.array(pred_lis).flatten()
            # cor = np.corrcoef(ori_np,pred_np)
            r2 = 1 - np.sum((ori_np - pred_np)**2) / np.sum((ori_np - np.mean(ori_np))**2)
            rou = scipy.stats.spearmanr(ori_np,pred_np)[0]

            plt.xlim(-2,4)
            plt.ylim(-2,4)
            plt.scatter(np.log10(ori_np),np.log10(pred_np),alpha=0.1,s=2)
            plt.scatter(np.log10(ori_np),np.log10(ori_np),s=0.1,color='red')
            # plt.title('corr:'+ str(cor[0][1]))
            plt.title('r2:'+ str(r2)+'\n'+'spearman:'+str(rou))
            plt.savefig(f'figs/{epoch}.png')
            plt.cla()

    def draw_spec(self):
        self.load_ckpt()
        self.model.eval()
        loader = self.test_loader
        with torch.no_grad():
            rou_lis =[]
            for i,data in enumerate(loader):

                # in_data = data[0].to(torch.float32).to(self.device)
                in_data = data[0].to(torch.float32).cuda()
                in_data = in_data.unsqueeze(1)
                # in_data = in_data.permute(0,2,1)
                output = self.model(in_data)
                # out_w = out_w.cpu().detach().numpy().tolist()
                # self.writer.write(str(out_w)+'\n')
                # ori_data = data[1].to(torch.float32).to(self.device)
                ori_data = data[1].to(torch.float32).cuda()
                ori_data = ori_data.unsqueeze(1)
                # ori_data.permute(0,2,1)
                output_np = np.array(output.cpu().squeeze())
                ori_data_np = np.array(ori_data.cpu().squeeze())
                noi_inp = np.array(data[0].squeeze())
                noi_inp = noi_inp[:1000]
                x = np.array([n for n in range(1000)])
                
                plt.figure(dpi=600)
                plt.plot(x,noi_inp,color='green',label='noised_data',linewidth=2)
                plt.plot(x,ori_data_np,color='blue',label='ori_data',linewidth=1.5)
                plt.plot(x,output_np,color='red',label='pred',linewidth=1)
                with open(f'spec_data/{i}.csv','w') as w:
                    w.write('idx,noi_inp,ori,output\n')
                    for idx in range(len(x)):
                        w.write(f'{x[idx]},{noi_inp[idx]},{ori_data_np[idx]},{output_np[idx]}\n')

                rou = scipy.stats.spearmanr(ori_data_np,output_np)[0]
                rou2 = scipy.stats.spearmanr(ori_data_np,noi_inp)[0]
                rou_lis.append(rou)
                plt.title('pred-ori spearman:'+str(rou)+'\n'+'noi-ori spearman' + str(rou2))

                plt.legend()
                plt.savefig(f'spec/{i}.png')
                plt.cla()
                plt.close()
            rou_np = np.array(rou_lis).mean()
            print(rou_np)
            with open('logs/spearman.log','a') as w:
                w.write(str(self.batchsz)+':\n')
                w.write(str(rou_lis)+'\n')



if __name__ =='__main__':
    # lsfn = Spearman_loss()
    aaa = torch.randn(32,1,100)
    # aaa_rank = aaa.argsort(dim=2).float()
    # print(aaa)
    # print(aaa_rank)
    # print(aaa_rank[0])
    # print(aaa_rank[1])
    bbb = torch.randn(32,1,100)
    # ls = lsfn(aaa,bbb)
    # # print(ls)
    # # spr = spearman_correlation(aaa,bbb)
    spr = spearmanr(aaa,bbb)
    print(spr)





