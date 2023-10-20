import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from Data_Loader4000 import FreqDataset, split,FreqModeDataset
import torch.nn as nn
import torch
from model.Unet4 import UNet
from torch.utils.data import Subset
import numpy as np
from trainer_para import Trainer
import argparse
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument("--a", default=0.001)
parser.add_argument("--b", default=64)
parser.add_argument("--local-rank", default=-1,type=int)
args = parser.parse_args()
# local_rank = args.local_rank
local_rank = int(os.environ["LOCAL_RANK"])
print(f"local rank: {local_rank}")

torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl",
                                     init_method='env://',
                                     world_size = 2,
                                     rank= local_rank)

device = torch.device("cuda", local_rank)
# print(device)


print('=========== Loading Dataset',local_rank,'===========')
data_set = FreqDataset(mode='IR',minmax=False)
train_idx,val_idx,test_idx = data_set.split(shuffle=True)   
train_set = Subset(data_set,train_idx)
val_set = Subset(data_set,val_idx)
test_set = Subset(data_set,test_idx)
print('========== Load Dataset Done',local_rank,'==========')
# print(test_set[0][0])
# print(test_set[0][1])

print('============ Loading Model',local_rank,'============')
model = UNet()

trainer = Trainer(model,train_set,val_set,test_set,batchsize=int(args.b),Lr=float(args.a),a=2,b=250,device=device,local_rank=local_rank)

print('=========== Start Training',local_rank,'===========')

# trainer.train(epochs=200)
if torch.distributed.get_rank() == 0:
    trainer.draw_pict(200)
    trainer.draw_spec()