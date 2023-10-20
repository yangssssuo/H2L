import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self,n_inp,n_hid,n_dep,dropout) -> None:
        super(TransformerEncoder,self).__init__()
        self.inp = nn.Linear(n_inp,n_hid)
        self.atten = nn.ModuleList([
            AttnBlk(dropout) for _ in range(n_dep)
        ])
        self.do = nn.Dropout(0.3)

    def forward(self,x):
        x = F.relu(self.inp(x))
        for layer in self.atten:
            x = x + self.do(layer(x))

class AttnBlk(nn.Module):
    def __init__(self,n_hid,dropout) -> None:
        super(AttnBlk,self).__init__()
        self.emb = Embedding(n_hid)
        self.attn = AttentionLayer(dropout)
    
    def forward(self,x):
        Q,K,V = self.emb(x)
        V_, o_w = self.attn(Q,K,V)
        return V_


class AttentionLayer(nn.Module):
    def __init__(self,dropout) -> None:
        super(AttentionLayer,self).__init__()
        self.softmax = nn.Softmax()
        self.do_layer = nn.Dropout(dropout)

    def forward(self,Q,K,V):
        #QKV: [bsz,length,word_vec_len]
        bsz = Q.shape[0]
        d_wvec = Q.shape[2]
        
        attention_w = torch.matmul(Q,K.permute(0,2,1)) / torch.sqrt(d_wvec).to(K.device)

        attention_w = self.do_layer(self.softmax(attention_w))

        V = torch.matmul(attention_w,V)

        return V, attention_w

class Embedding(nn.Module):
    def __init__(self,n_hid) -> None:
        super(Embedding,self).__init__()
        self.w_q = nn.Linear(n_hid,n_hid)
        self.w_k = nn.Linear(n_hid,n_hid)
        self.w_v = nn.Linear(n_hid,n_hid)

    def forward(self,x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        return Q,K,V




# aaa = torch.randn(32,24,1)
# bbb = torch.randn(32,1,35)
# xxx = torch.randn(32,35,1)

# ccc = torch.matmul(aaa,bbb)
# ddd = torch.matmul(ccc,xxx)
# print(ddd.shape)
