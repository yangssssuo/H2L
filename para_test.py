import torch
from model.test import Net
from Data_Loader import FreqDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from torchvision import datasets
from torchvision.transforms import ToTensor

local_rank = int(os.environ["LOCAL_RANK"])

device = torch.device("cuda", local_rank)

torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl",
                                     init_method='env://',
                                     world_size = 2,
                                     rank= local_rank)

print('=========== Loading Dataset',local_rank,'===========')

data_set = datasets.MNIST(
    root="./mnist",
    train=True,
    download=False,
    transform=ToTensor()
)
# data_set = FreqDataset(mode='IR',minmax=False)
train_sampler = DistributedSampler(data_set)
dataloader = DataLoader(data_set,batch_size=512,sampler=train_sampler)
print('========== Load Dataset Done',local_rank,'==========')

print('============ Loading Model',local_rank,'============')
net = Net()
net = net.to(device=device)
# print('a')
model = torch.nn.parallel.DistributedDataParallel(net,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
print('=========== Load Model Done',local_rank,'===========')

for epoch in range(10):
    dataloader.sampler.set_epoch(epoch)
    for data in dataloader:
        # print(data.shape)
        aaa = model(data)
        print(aaa.shape)
    # output = model(data)
    # print(output.shape)
