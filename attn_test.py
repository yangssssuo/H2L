from Data_Loader import FreqDataset, split,FreqModeDataset
import torch.nn as nn
import torch
# device = torch.device("cuda:2")
# from model.conv1d import Net
from model.attn import SpecAttn
from torch.utils.data import Subset
import numpy as np
from trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hid", type=int, default=3600, help="hidden size of transformer model")
parser.add_argument("--batch_size", type=int, default=256)
# parser.add_argument("--epochs", type=int, default=500)
# parser.add_argument("--lr", type=float, default=0.00005)
# parser.add_argument('--heads', default=1, type=int)
args = parser.parse_args()

data_set = FreqDataset()

train_idx,val_idx,test_idx = data_set.split(shuffle=True)   

train_set = Subset(data_set,train_idx)
val_set = Subset(data_set,val_idx)
test_set = Subset(data_set,test_idx)

model = SpecAttn(args.hid,4001,1000)

trainer = Trainer(model,train_set,val_set,test_set,batchsize=args.batch_size)

trainer.train()

trainer.draw_pict(500)

trainer.draw_spec()