from Data_Loader import FreqDataset, split,FreqModeDataset
import torch.nn as nn
import torch
from model.self_attn import SpecAttn,DeepAttn
from torch.utils.data import Subset
import numpy as np
from trainer import Trainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hid", type=int, default=1000, help="hidden size of transformer model")
# parser.add_argument("--batch_size", type=int, default=256)
# parser.add_argument("--epochs", type=int, default=500)
# parser.add_argument("--lr", type=float, default=0.00005)
parser.add_argument('--heads', default=1, type=int)
parser.add_argument('--dep', default=3, type=int)
args = parser.parse_args()

data_set = FreqDataset(mode='Raman')

train_idx,val_idx,test_idx = data_set.split(shuffle=True)   

train_set = Subset(data_set,train_idx)
val_set = Subset(data_set,val_idx)
test_set = Subset(data_set,test_idx)

model = DeepAttn(4000,args.hid,args.heads,1000,args.dep)

trainer = Trainer(model,train_set,val_set,test_set,batchsize=128,Lr=0.001)

trainer.train(epochs=1000)

trainer.draw_pict(1000)

trainer.draw_spec()