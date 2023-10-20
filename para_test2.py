import torch
from model.test import Net
from Data_Loader import FreqDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from torchvision import datasets
from torchvision.transforms import ToTensor
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



print('=========== Loading Dataset===========')

data_set = datasets.MNIST(
    root="./mnist",
    train=True,
    download=False,
    transform=ToTensor()
)
# data_set = FreqDataset(mode='IR',minmax=False)

dataloader = DataLoader(data_set,batch_size=512)
print('========== Load Dataset Done==========')

print('============ Loading Model============')
net = Net()
net = net.cuda()
# print('a')
model = torch.nn.parallel.DataParallel(net)
print('=========== Load Model Done===========')

for data in dataloader:
    # print(data[0].shape)
    aaa = model(data[0])
    print(aaa.shape)
