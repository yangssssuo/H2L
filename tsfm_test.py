from Data_Loader2000 import FreqModeDataset
from torch.utils.data import Dataset, DataLoader
aaa = FreqModeDataset()
# print(len(aaa))
# print(aaa[0][0].shape)
# print(aaa[0][1].shape)

dataloader = DataLoader(aaa,batch_size=32)
for i,data in enumerate(dataloader):
    print(data[0].shape)
    print(data[1].shape)