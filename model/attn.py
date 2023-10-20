import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class SpecAttn(nn.Module):
    def __init__(self,n_hid,n_inp,n_out,dropout=0.5) -> None:
        super(SpecAttn,self).__init__()
        self.hid_dim = n_hid
        self.dropout = dropout

        self.w_q = nn.Linear(n_inp,n_hid)
        self.w_k = nn.Linear(n_inp,n_hid)
        self.w_v = nn.Linear(n_inp,n_hid)

        self.fc = nn.Linear(n_hid,n_out)
        self.do_layer = nn.Dropout(dropout)
        # self.scale = torch.sqrt(torch.FloatTensor(4000))

    def forward(self,x):
        '''
        x:[batch_sz, length of freq mode, hidden dim]
        '''
        bsz = x.shape[0]
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)


        energy = torch.matmul(Q,K.permute(0,2,1)) / torch.sqrt(torch.tensor(self.hid_dim))

        attention = self.do_layer(F.softmax(energy,dim=-1))

        x = torch.matmul(attention,V)
        x = self.fc(x)

        return x