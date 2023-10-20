import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import scipy.stats
import seaborn as sns
import os
from torch.utils.data.distributed import DistributedSampler

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

# torch.distributed.init_process_group(backend="nccl")

def earth_mover_distance(y_true, y_pred):
    aaa =  torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)
    return aaa.mean()


class Trainer:
    def __init__(self,model,train_set,valid_set,test_set,device,local_rank,batchsize=64,num_workers=16,Lr=0.01,a=1,b=100) -> None:
        # self.device = torch.device("cuda:3")
        # self.model = model.to(self.device)
        # f = open('/home/yanggk/Data/HW/attn.log','a')
        # self.writer = f
        print("start init")
        self.device = device
        # model = model.to(self.device)
        model = model.to(local_rank)
        print("model to device ed.")
        # self.local_rank = torch.distributed.get_rank()
        self.model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
        # self.model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        print("model to GPU ed.")
        self.a = a
        self.b = b
        self.batchsz = batchsize
        print("train data loader start")
        self.train_sampler = DistributedSampler(train_set)
        # self.train_loader = DataLoader(train_set,batch_size=batchsize,num_workers=num_workers)
        self.train_loader = DataLoader(train_set,batch_size=batchsize,sampler=self.train_sampler)
        print("train data loader ed.")
        self.valid_loader = DataLoader(valid_set)
        self.test_loader = DataLoader(test_set)
        # self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.loss_fn = torch.nn.L1Loss(reduction='mean')
        self.loss_fn2 = torch.nn.CosineEmbeddingLoss()
        # self.loss_fn = earth_mover_distance
        # self.loss_fn = torch.nn.Spearman_loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=Lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=80,min_lr=1e-06,verbose=True)
        self.records = {'val_losses': []}
        print("finish init")
        
    def train_iterations(self,mode='train'):
        if mode =='transport':
            self.load_ckpt()
        self.model.train()
        losses = []
        attnn = torch.zeros(124,124).to(self.device)
        len_loader = len(self.train_loader) - 1 
        for i,data in enumerate(self.train_loader):
            # data = Variable(data)
            tgt = torch.ones(1).to(self.device)
            self.optimizer.zero_grad()
            # in_data = data[0].to(torch.float32).to(self.device)
            in_data = Variable(data[0].to(torch.float32).to(self.device))
            in_data = in_data.unsqueeze(1)
            output,attn_w = self.model(in_data)
            attn_w = torch.sum(attn_w,dim=0)
            attnn = attnn + attn_w
            # ori_data = data[1].to(torch.float32).to(self.device)
            ori_data = Variable(data[1].to(torch.float32).to(self.device))
            ori_data = ori_data.unsqueeze(1)
            loss = self.loss_fn(output,ori_data)* self.a + self.loss_fn2(output.squeeze(),ori_data.squeeze(),tgt) * self.b
            # loss = self.loss_fn2(output.squeeze(),ori_data.squeeze(),tgt)
            # loss = self.compute_rank_correlation(output,ori_data)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.cpu().detach())
            # if i == len_loader:
            #     save_attn_w = torch.sum(attn_w,dim=0).detach().cpu().numpy()
            #     np.savetxt('/home/yanggk/Data/H2L_Data/attn.txt',save_attn_w,delimiter=',')

        # self.scheduler.step(loss)
        attnn = attnn.cpu().detach()
        trn_loss = np.array(losses).mean()
        return trn_loss,attnn


    def valid_iterations(self,mode='valid'):
        self.model.eval()
        if mode == 'test': loader = self.test_loader
        if mode == 'valid': loader = self.valid_loader
        losses = []
        with torch.no_grad():
            for i,data in enumerate(loader):
                # data = Variable(data)
                tgt = torch.ones(1).to(self.device)
                # in_data = data[0].to(torch.float32).to(self.device)
                in_data = Variable(data[0].to(torch.float32).to(self.device))
                in_data = in_data.unsqueeze(1)
                output,_ = self.model(in_data)
                # ori_data = data[1].to(torch.float32).to(self.device)
                ori_data = Variable(data[1].to(torch.float32).to(self.device))
                ori_data = ori_data.unsqueeze(1)
                loss = self.loss_fn(output,ori_data)* self.a + self.loss_fn2(output.squeeze(0),ori_data.squeeze(0),tgt) * self.b
                # loss = self.loss_fn2(output.squeeze(0),ori_data.squeeze(0),tgt)
                # loss = self.compute_rank_correlation(output,ori_data)
                losses.append(loss.cpu().detach())
        val_loss = np.array(losses).mean()
        return val_loss



    def train(self,epochs=500,mode='train'):
        for epoch in range(epochs):
            train_loss,attnn = self.train_iterations(mode)
            val_loss = self.valid_iterations()
            self.records['val_losses'].append(val_loss)
            self.scheduler.step(val_loss)
            save_log = ''
            if val_loss == np.array(self.records['val_losses']).min():
                save_log = 'save best model'
                self.save_model()
            torch.distributed.barrier()
            print('epoch: {} train_loss: {} val_loss: {}, {}'.format(epoch, train_loss, val_loss, save_log))
            # self.test_model(epoch)
            if torch.distributed.get_rank() == 0:
                if epoch % 100 == 0:
                    self.draw_pict(epoch)
                    self.draw_hot(attnn,epoch)
                if epoch == epochs-1:
                    self.draw_hot(attnn,epoch)
            else: pass

        self.load_ckpt('best_model.ckpt')
        test_loss = self.valid_iterations(mode='test')
        print('test_loss: ', test_loss)

    def draw_hot(self,data,epoch):
        plt.figure(figsize=(20,20),dpi=600)
        sns.heatmap(data,xticklabels=False, yticklabels=False,square=True)
        plt.savefig(f'figs/heat{epoch}.png')
        plt.close()

    def save_model(self):
        file_name = 'best_model.ckpt'
        if torch.distributed.get_rank() == 0:
            with open(os.path.join(file_name),'wb') as f:
                torch.save({'model_state_dict': self.model.state_dict()}, f)
        else: pass

    def load_ckpt(self, ckpt_path='best_model.ckpt'):
        ckpt = torch.load(ckpt_path)
        # torch.distributed.barrier()
        self.model.load_state_dict(ckpt['model_state_dict'])
        

    def draw_pict(self,epoch=100):
        self.load_ckpt()
        self.model.eval()
        loader = self.test_loader
        with torch.no_grad():
            ori_lis = []
            pred_lis = []
            for i,data in enumerate(loader):
                # data = Variable(data)
                # in_data = data[0].to(torch.float32).to(self.device)
                in_data = Variable(data[0].to(torch.float32).to(self.device))
                in_data = in_data.unsqueeze(1)
                # in_data = in_data.permute(0,2,1)
                output,_= self.model(in_data)
                # ori_data = data[1].to(torch.float32).to(self.device)
                ori_data = Variable(data[1].to(torch.float32).to(self.device))
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

            # plt.xlim(-2,4)
            # plt.ylim(-2,4)
            # plt.scatter(np.log10(ori_np),np.log10(pred_np),alpha=0.1,s=2)
            # plt.scatter(np.log10(ori_np),np.log10(ori_np),s=0.1,color='red')
            plt.scatter(ori_np,pred_np,alpha=0.1,s=2)
            plt.scatter(ori_np,ori_np,s=0.1,color='red')
            # plt.title('corr:'+ str(cor[0][1]))
            plt.title('r2:'+ str(r2)+'\n'+'spearman:'+str(rou))
            plt.savefig(f'figs/{torch.distributed.get_rank()}_{epoch}.png')
            plt.cla()

    def draw_spec(self):
        self.load_ckpt()
        self.model.eval()
        loader = self.test_loader
        with torch.no_grad():
            rou_lis =[]
            attn_lis = []
            attn = torch.zeros(128,128)
            for i,data in enumerate(loader):
                # data = Variable(data)
                # in_data = data[0].to(torch.float32).to(self.device)
                in_data = Variable(data[0].to(torch.float32).to(self.device))
                in_data = in_data.unsqueeze(1)
                # in_data = in_data.permute(0,2,1)
                output,attn_w = self.model(in_data)
                # out_w = out_w.cpu().detach().numpy().tolist()
                # self.writer.write(str(out_w)+'\n')
                # ori_data = data[1].to(torch.float32).to(self.device)
                ori_data = Variable(data[1].to(torch.float32).to(self.device))
                ori_data = ori_data.unsqueeze(1)
                # ori_data.permute(0,2,1)
                output_np = np.array(output.cpu().squeeze())
                ori_data_np = np.array(ori_data.cpu().squeeze())
                ori_data_np = ori_data_np[:1000]
                noi_inp = np.array(data[0].squeeze())
                noi_inp = noi_inp[:1000]
                output_np = output_np[:1000]
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
                plt.title('pred-ori spearman:'+str(rou)+'\n'+'noi-ori spearman:' + str(rou2))

                plt.legend()
                plt.savefig(f'spec/{i}.png')
                plt.cla()
                plt.close()

                # attn_list = attn_w.squeeze().detach().cpu().numpy().tolist()
                # sns.heatmap(attn_list,xticklabels=False, yticklabels=False)
                # plt.savefig(f'attn_w/{i}.png')
                # plt.cla()

                attn_sum = torch.sum(attn_w.squeeze(),dim=1).detach().cpu().numpy().tolist()
                attn_lis.append(attn_sum)
                xs = [xxx for xxx in range(len(attn_sum))]
                plt.plot(xs,attn_sum)
                plt.savefig(f'attn_w/{i}.png')
                plt.cla()
                plt.close()
            attnww = np.array(attn_lis).sum(axis=0)
            # attnww2 = np.array(attn_lis).sum(axis=1)
            plt.plot(xs,attnww)
            plt.savefig('attn.png')
            plt.close()
            # plt.plot(xs,attnww2)
            # plt.savefig('attn2.png')
            # plt.close()
            rou_np = np.array(rou_lis).mean()
            print(rou_np)
            with open('logs/spearman.log','a') as w:
                w.write(str(self.batchsz)+':\n')
                w.write(str(rou_lis)+'\n')



if __name__ =='__main__':
    aaa = torch.randn(32,1,1000)
    bbb = torch.randn(32,1,1000)
    ccc = aaa + bbb
    print(ccc.shape)

    zzz = torch.zeros(32,128,128)
    print(torch.sum(zzz,dim=0).shape)

    # dis = earth_mover_distance(aaa,bbb)

    # l1_loss = nn.L1Loss(reduction='mean')

    # l1 = l1_loss(aaa,bbb)

    # print(dis)

    # print(l1)



