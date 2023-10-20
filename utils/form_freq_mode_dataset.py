import pandas as pd
import numpy as np
from pyparsing import line
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random

class FreqModeDataset(Dataset):
    def __init__(self,tsv_file_path='/home/yanggk/Data/HW/tsv/') -> None:
        super(FreqModeDataset).__init__()
        self.tsv_files = tsv_file_path

        data =[]

        for idx in range(1,3501):
            if idx == 1355:pass
            elif idx == 2607:pass
            elif idx == 3039:pass
            else:
                a = pd.read_csv(f'{self.tsv_files}{idx}.tsv',delimiter='\t',index_col=0)
                data.append(a)
        
        data = np.array(data)
        low_part = data[:,:35,:]
        high_part = data[:,35:,:]

        ls_low = []
        for n,line in enumerate(low_part):
            line = line.tolist()
            line = self.make_noise(line)
            ls_low.append(line)
        ls_low = np.array(ls_low)
        # print(ls_low-low_part)
        noi_data_L = torch.from_numpy(ls_low)
        high_part = torch.from_numpy(high_part)
        self.noi_data = torch.cat((noi_data_L,high_part),dim=1)

        
        self.mode_data = torch.from_numpy(data)

    @staticmethod
    def make_noise(item):
        mu =0 
        sigma = 0.05
        for i in range(len(item)):
            for n in range(len(item[i])):
                item[i][n] += random.gauss(mu,sigma)
        return item

aaa= FreqModeDataset()
print(aaa.mode_data.shape)
print(aaa.noi_data.shape)