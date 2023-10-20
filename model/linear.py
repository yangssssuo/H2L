import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class LinNet(nn.Module):
    def __init__(self,n_in,n_dep,n_hid,n_out) -> None:
        super(LinNet,self).__init__()
        self.input_f = n_in
        self.output_f = n_out
        self.n_dep = n_dep
        self.n_hid = n_hid

        self.inp = nn.Linear(n_in,n_hid)

        self.lin_blk = nn.ModuleList([
            nn.Linear(self.n_hid,self.n_hid) for _ in range(self.n_dep)
        ])

        self.outp = nn.Linear(n_hid,n_out)

        # self.batchnorm = nn.BatchNorm1d(n_in)


    def forward(self,x):
        # x = x.permute(0,2,1)
        # x = self.batchnorm(x)
        # x = x.permute(0,2,1)

        x = F.relu(self.inp(x))
        for lin_layer in self.lin_blk:
            x = F.relu(lin_layer(x))
        
        x = self.outp(x)

        return x

