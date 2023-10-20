from Data_Loader import FreqDataset, split,FreqModeDataset
import torch.nn as nn
import torch
from model.Unet2 import UNet
from torch.utils.data import Subset
import numpy as np
from trainer import Trainer
import argparse
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import os

parser = argparse.ArgumentParser()
parser.add_argument("--a", default=0.001)
parser.add_argument('--b', default=64)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_set = FreqDataset(mode='IR',minmax=False)

train_idx,val_idx,test_idx = data_set.split(shuffle=True)   

train_set = Subset(data_set,train_idx)
val_set = Subset(data_set,val_idx)
test_set = Subset(data_set,test_idx)

print(test_idx)

# print(test_set[0][0])
# print(test_set[0][1])

model = UNet()

trainer = Trainer(model,train_set,val_set,test_set,batchsize=int(args.b),Lr=float(args.a),a=2,b=250)
# trainer.train(epochs=200)
trainer.draw_pict(200)
trainer.draw_spec()