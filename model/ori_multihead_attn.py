from turtle import forward
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class ConvLinBlock(nn.Module):
    def __init__(self) -> None:
        super(ConvLinBlock,self).__init__()
        self.conv = nn.Conv1d(1,1,10,4)
        self.lin = nn.Linear(998,1000)

    def forward(self,x):
        x = self.conv(x)
        x = self.lin(x)
        x = F.relu(x)
        return x

class Net(nn.Module):
    def __init__(self,hid,outp,dropout) -> None:
        super(Net,self).__init__()
        self.w_q = ConvLinBlock()
        self.w_k = ConvLinBlock()
        self.w_v = ConvLinBlock()
        self.mul_h_attn = nn.MultiheadAttention(1,1,dropout)
        self.outp = nn.Linear(hid,outp)
        self.relu = nn.ReLU()
        # f = open('/home/yanggk/Data/HW/bkup/attns/attn.log','a')
        # self.writer = f

    def forward(self,x):
        Q = F.relu(self.w_q(x)).permute(2,0,1)
        K = F.relu(self.w_k(x)).permute(2,0,1)
        V = F.relu(self.w_v(x)).permute(2,0,1)

        x, out_w = self.mul_h_attn(Q,K,V)
        x = x.permute(1,2,0)
        # torch.save(out_w,'/home/yanggk/Data/HW/bkup/attns/attn_w.pth')
        # out_w = out_w.cpu().detach().numpy().tolist()
        # self.writer.write(str(out_w))
        x = self.outp(x)
        return x,out_w

class MultiNet(nn.Module):
    def __init__(self) -> None:
        super(MultiNet,self).__init__()
        self.batchnorm = nn.BatchNorm1d(4001)
        self.blk = Net(1000,1000,0.3)

    def forward(self,x):
        # x = x.permute(0,2,1)
        # x = self.batchnorm(x)
        # x = x.permute(0,2,1)
        x = self.blk(x)
        return x