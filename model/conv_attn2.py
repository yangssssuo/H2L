import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class ConvAttn(nn.Module):
    def __init__(self) -> None:
        super(ConvAttn,self).__init__()
        self.batchnorm = nn.BatchNorm1d(4001)
        self.w_q = ConvLinBlock()
        self.w_k = ConvLinBlock()
        self.w_v = ConvLinBlock()
        self.attn = nn.MultiheadAttention(1,1,0.3)
        self.out = nn.Linear(1000,1000)

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.batchnorm(x)
        x = x.permute(0,2,1)
        Q = self.w_q(x).permute(2,0,1)
        K = self.w_k(x).permute(2,0,1)
        V = self.w_v(x).permute(2,0,1)
        out, out_w = self.attn(Q,K,V)
        out = out.permute(1,2,0)
        out = self.out(out)
        return out


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
        

