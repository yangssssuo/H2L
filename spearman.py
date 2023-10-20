import numpy as np
import torch
from torchmetrics import SpearmanCorrCoef

def compute_rank_correlation(x, y):
    """
    x,y :[bsz,1,length]
    """
    batch_sz = x.shape[0]
    x_mean = x.mean(2).unsqueeze(1)
    y_mean = y.mean(2).unsqueeze(1)
    d_x = x - x_mean
    d_y = y - y_mean
    upper = torch.sum(d_x * d_y,dim=2)
    downer = torch.sum(torch.square(d_x) * torch.square(d_y),dim=2)
    spear = upper/downer
    spearman = spear.mean()
    return torch.tensor(1) - spearman

def spearman(x,y):
    dxy = x - y
    d2_sum = torch.square(dxy).sum(dim=2)
    spear = torch.tensor(6) * d2_sum / torch.tensor(1000 * 99999)

    return torch.tensor(1) - spear 


aaa = torch.rand(32,1,1000)
bbb = torch.rand(32,1,1000)

print(spearman(aaa,bbb))
# a_mean = aaa.mean(2).unsqueeze(1)
# print(a_mean.shape)
# ccc = aaa-a_mean
# print(ccc.shape)
# sqr = torch.square(ccc)
# print(sqr)
# print(sqr.shape)
# # ddd = ccc * ccc
# # print(ddd)
# # print(ddd.shape)
# # eee = ddd.sum(dim=2)
# # print(eee)
# # print(eee.shape)