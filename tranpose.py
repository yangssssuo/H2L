from Data_Loader import FreqDataset, split,FreqModeDataset
import torch.nn as nn
import torch
from model.GRU import UNet
from torch.utils.data import Subset
import numpy as np
from trainer import Trainer
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a", type=int, default=0)
parser.add_argument('--b', default=1, type=int)
args = parser.parse_args()

data_set = FreqDataset(csv_path="/home/yanggk/Data/H2L_Data/tsps/Au-50/IR/ts_formed_1cm-1_clr.csv",noi_path="/home/yanggk/Data/H2L_Data/tsps/Au-50/IR/ts_formed_1cm-1_noi.csv",mode='IR')

# print(data_set[0])
train_idx,val_idx,test_idx = data_set.split(shuffle=True)   

train_set = Subset(data_set,train_idx)
val_set = Subset(data_set,val_idx)
test_set = Subset(data_set,test_idx)
print(len(train_set))
print(len(val_set))
print(len(test_set))
model = UNet()

trainer = Trainer(model,train_set,val_set,test_set,batchsize=64,Lr=0.001,a=2,b=250)
trainer.train(epochs=200,mode='transport')
trainer.draw_pict(200)
trainer.draw_spec()