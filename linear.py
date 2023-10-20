from Data_Loader import FreqDataset, split,FreqModeDataset
import torch.nn as nn
import torch
from model.linear import LinNet
from torch.utils.data import Subset
import numpy as np
from trainer_bsl import Trainer
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a", type=int, default=1, help="hidden size of transformer model")
parser.add_argument('--b', default=100, type=int)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_set = FreqDataset(mode='Raman')

train_idx,val_idx,test_idx = data_set.split(shuffle=True)   

train_set = Subset(data_set,train_idx)
val_set = Subset(data_set,val_idx)
test_set = Subset(data_set,test_idx)

# print(test_set[0][0])
# print(test_set[0][1])

model = LinNet(4000,5,4000,1000)

trainer = Trainer(model,train_set,val_set,test_set,batchsize=64,Lr=0.001)
trainer.train(epochs=200)
trainer.draw_pict(200)
trainer.draw_spec()