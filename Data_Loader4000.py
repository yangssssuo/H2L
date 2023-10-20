import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random

def get_data(file_path):
    aaa = pd.read_csv(file_path,header=None)

    IR = aaa.iloc[:,1:4002]
    Raman = aaa.iloc[:,4002:]
    
    IR = torch.from_numpy(IR.values)
    # print(IR.shape)

    Raman = torch.from_numpy(Raman.values)
    # print(Raman.shape)

class FreqDataset(TensorDataset):
    def __init__(self, dataset='Freqset',csv_path="/home/yanggk/Data/H2L_Data/formed_1cm-1.csv",mode='Raman',noi_path="/home/yanggk/Data/H2L_Data/formed_1cm-1_noised.csv",minmax=False) -> None:
        super(FreqDataset,self).__init__()
        self.mode = mode
        self.minmax = minmax
        self.dataset = dataset
        self.datalist_path = csv_path
        self.noi_path = noi_path
        aaa = pd.read_csv(self.datalist_path,header=None)
        bbb = pd.read_csv(self.noi_path,header=None)

        clear_IR = aaa.iloc[:,1:4001]
        clear_Raman = aaa.iloc[:,4001:]

        wide_IR = bbb.iloc[:,1:4001]
        wide_Raman = bbb.iloc[:,4001:]

        clear_IR = torch.from_numpy(clear_IR.values)
        clear_Raman = torch.from_numpy(clear_Raman.values)

        wide_IR = torch.from_numpy(wide_IR.values)
        wide_Raman = torch.from_numpy(wide_Raman.values)

        L_wide_IR = wide_IR[:,:1000]
        L_wide_IR = self.make_noise(L_wide_IR)
        L_wide_Raman = wide_Raman[:,:1000]
        L_wide_Raman = self.make_noise(L_wide_Raman)

        self.L_IR = clear_IR[:,:1000]
        H_IR = clear_IR[:,1000:]

        self.L_Raman = clear_Raman[:,:1000]
        H_Raman = clear_Raman[:,1000:]

        noi_IR = np.concatenate((L_wide_IR,H_IR),axis=1)
        noi_Raman = np.concatenate((L_wide_Raman,H_Raman),axis=1)
        self.noi_IR = torch.from_numpy(noi_IR)
        self.noi_Raman = torch.from_numpy(noi_Raman)

        self.clear_IR = clear_IR
        self.clear_Raman = clear_Raman

    def min_max(self,t):
        min_t = torch.min(t)
        max_t = torch.max(t)
        return (t - min_t) / (max_t-min_t)


    def split(self, shuffle=False):

        idx = np.arange(0, len(self.noi_Raman))
        if shuffle:
            np.random.shuffle(idx)
        train_idx = idx[:int(0.8*len(self.noi_Raman))]
        valid_idx = idx[int(0.8*len(self.noi_Raman)):int(0.9*len(self.noi_Raman))]
        test_idx = idx[int(0.9*len(self.noi_Raman)):]

        return train_idx, valid_idx, test_idx


    def __len__(self):
        if self.mode == 'Raman':
            return len(self.noi_Raman)
        if self.mode == 'IR':
            return len(self.noi_IR)

    def __getitem__(self, idx):
        if self.mode == 'Raman':
            noi_data = self.noi_Raman[idx]
            L_data = self.clear_Raman[idx]
            # return noi_data,L_data
        elif self.mode == 'IR':
            noi_data = self.noi_IR[idx]
            L_data = self.clear_IR[idx]
            # return noi_data,L_data
        else: pass

        if self.minmax:
            noi_data = self.min_max(noi_data)
            L_data = self.min_max(L_data)
        else: pass

        return noi_data,L_data

    def make_noise(self,item):
        mu =0.0
        if self.mode == 'Raman':
            sigma = 20.0
            cutoff = 80.0
        else: 
            sigma = 5.0
            cutoff = 20.0
        noise = torch.normal(mu,sigma,(item.shape[0],item.shape[1]))
        for i in range(item.shape[0]):
            for j in range(item.shape[1]):
                if item[i][j] <= cutoff:
                    item[i][j] += noise[i][j]
                else: pass
        return torch.nn.functional.relu(item)


# aaa = FreqDataset()

# print(aaa.L_IR)

def split(dataloader, batch_size=1, split=0.8):
    
    """Splits the given dataset into training/validation.
       Args:
           dataset[torch dataloader]: Dataset which has to be split
           batch_size[int]: Batch size
           split[float]: Indicates ratio of validation samples
       Returns:
           train_set[list]: Training set
           val_set[list]: Validation set
    """

    index = 0
    length = len(dataloader)

    train_set = []
    val_set = []

    for data, target in dataloader:
        if index <= (length * split):
            train_set.append([data, target])
        else:
            val_set.append([data, target])

        index += 1

    return train_set, val_set
    
class FreqModeDataset(Dataset):
    def __init__(self,tsv_file_path='/home/yanggk/Data/HW/tsv/') -> None:
        super(FreqModeDataset,self).__init__()
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
        self.L_data = torch.from_numpy(low_part)
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

    def split(self, shuffle=False):

        idx = np.arange(0, len(self.mode_data))
        if shuffle:
            np.random.shuffle(idx)
        train_idx = idx[:int(0.8*len(self.mode_data))]
        valid_idx = idx[int(0.8*len(self.mode_data)):int(0.9*len(self.mode_data))]
        test_idx = idx[int(0.9*len(self.mode_data)):]

        return train_idx, valid_idx, test_idx

    @staticmethod
    def make_noise(item):
        mu =0 
        sigma = 0.05
        for i in range(len(item)):
            for n in range(len(item[i])):
                item[i][n] += random.gauss(mu,sigma)
        return item

    def __len__(self):
        return len(self.mode_data)

    def __getitem__(self, idx):
        noi_data = self.noi_data[idx]
        L_data = self.L_data[idx]
        return noi_data,L_data
