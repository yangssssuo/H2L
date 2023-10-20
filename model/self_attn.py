import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class DeepAttn(nn.Module):
    def __init__(self,n_inp,n_hid,n_heads,n_out,dep,dropout=0.3) -> None:
        super(DeepAttn,self).__init__()
        self.inp = nn.Linear(n_inp,n_hid)
        self.blks = nn.ModuleList([
            SpecAttn(n_hid,n_hid,n_heads,n_hid) for _ in range(dep)
        ])
        self.batchnorm = nn.BatchNorm1d(n_hid)
        self.outp = nn.Linear(n_hid,n_out)
        self.do = nn.Dropout(dropout)

    def forward(self,x):
        x = F.relu(self.inp(x))
        for layer in self.blks:
            x = x + F.relu(self.do(layer(x)))
        x = self.outp(x)
        return(x)

class SpecAttn(nn.Module):
    def __init__(self,n_hid,n_inp,n_heads,n_out,dropout=0.3) -> None:
        super(SpecAttn,self).__init__()
        self.hid_dim = n_hid
        self.dropout = dropout
        self.n_heads = n_heads
        f = open('logs/attn.log','a')
        self.writer = f
        assert n_hid % n_heads == 0

        self.w_q = nn.Linear(n_inp,n_hid)
        self.w_k = nn.Linear(n_inp,n_hid)
        self.w_v = nn.Linear(n_inp,n_hid)

        self.fc = nn.Linear(n_hid,n_out)
        self.do_layer = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([n_hid // n_heads]))

    def forward(self,x):
        '''
        x:[batch_sz, length of freq mode, hidden dim]
        '''
        bsz = x.shape[0]
        Q = self.w_q(x)

        K = self.w_k(x)
        V = self.w_v(x)
        '''
        [1,1,72]
        '''

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,3,1)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,3,1)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,3,1)
        '''
        view[1,1,4,18]
        permute[1,4,1,18]
        '''

        energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale.to(K.device)

        attention = F.softmax(energy,dim=-1)

        attention = self.do_layer(attention)

        # np.savetxt('logs/attn.csv',attention.detach().cpu().numpy(),delimiter=',')

        x = torch.matmul(attention,V)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(bsz,-1,self.n_heads * (self.hid_dim // self.n_heads))
        # np.savetxt('logs/attn.csv',x.cpu().numpy,delimiter=',')
        # x_lis = []
        # for i in range(3600):
        #     x_lis.append(str(x[0][0][i].cpu().numpy()))
        # self.writer.write(','.join(x_lis)+ '\n')
        x = self.fc(x)

        return x

class ConvLinBlock(nn.Module):
    def __init__(self,n_hid) -> None:
        super(ConvLinBlock,self).__init__()
        self.conv = nn.Conv1d(1,1,12,2)
        self.lin = nn.Linear(1995,n_hid)

    def forward(self,x):
        x = self.conv(x)
        x = self.lin(x)
        x = F.relu(x)
        return x