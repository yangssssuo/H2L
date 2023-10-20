from Data_Loader import FreqDataset, split,FreqModeDataset
import torch.nn as nn
import torch
# from model.conv1d import Net
from model.conv_attn2 import ConvAttn
from torch.utils.data import Dataset, DataLoader, TensorDataset,Subset
import numpy as np
from trainer import Trainer
import os
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# data_set = FreqDataset()
# #shape [4001,]

# train_idx,val_idx,test_idx = data_set.split(shuffle=True)   

# train_set = Subset(data_set,train_idx)
# val_set = Subset(data_set,val_idx)
# test_set = Subset(data_set,test_idx)

# # a = torch.rand(32,72,3)

# model = ConvAttn()

# trainer = Trainer(model,train_set,val_set,test_set,batchsize=len(train_set))

# # in data shape[bsz,1,4001]

# trainer.train()

# trainer.draw_pict(1000)

# trainer.draw_spec()

# a = torch.randn((32,1,4001))
# conv = nn.Conv1d(1,1,10,4)
# a = conv(a)
# print(a.shape)
o = torch.randn(1000,1000)
print(o)
a = torch.normal(0.0,1.0,(1000,1000))
for i in range(o.shape[0]):
    for j in range(o.shape[1]):
        if o[i][j] <= 0.5:
            o[i][j] = o[i][j] + a[i][j]
print(o)