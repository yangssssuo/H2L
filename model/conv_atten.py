import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


# class Conv3to1(nn.Mudule):
#     def __init__(self) -> None:
#         super(Conv3to1,self).__init__()
#         self.conv = nn.Conv1d(in_channels=3,out_channels=1,kernel_size=3)

#     def forward(self,x):
#         x = self.conv(x)

#         return x

class ConvWithAttn(nn.Module):
    def __init__(self) -> None:
        super(ConvWithAttn,self).__init__()
        self.conv = nn.Conv1d(3,1,1)
        self.attn = SelfAttn(72,72,4,35,0.3)
        self.reconv = nn.ConvTranspose1d(1,3,1)

    def forward(self,x):
        x = self.conv(x)
        # x = x.squeeze()
        # x = x.unsqueeze(0).unsqueeze(0)
        # x = x.permute(2,1,0)
        out = self.attn(x)
        out = self.reconv(out)

        return out


class SelfAttn(nn.Module):
    def __init__(self,n_hid,n_inp,n_heads,n_out,dropout=0.3) -> None:
        super(SelfAttn,self).__init__()
        self.hid_dim = n_hid
        self.dropout = dropout
        self.n_heads = n_heads

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

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)
        '''
        view[1,1,4,18]
        permute[1,4,1,18]
        '''

        energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale.to(K.device)

        attention = self.do_layer(F.softmax(energy,dim=-1))

        x = torch.matmul(attention,V)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(bsz,-1,self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x


