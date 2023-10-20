import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import torch

# aaa = pd.read_csv('logs/attn1025.log',header=None)
# for i in range(350):
#     data = aaa.iloc[i:i+1,:]
#     # print(data)
#     plt.figure(figsize=(100,4))
#     sns.heatmap(data,cbar_kws={'orientation': 'horizontal',"shrink": 0.3},
#                xticklabels=False, yticklabels=False)
#     plt.savefig(f'heatmap/{i}.png')
#     plt.cla()

# print(aaa[0][0])
# with open('logs/attn1025.log','r') as f:
#     aaa = f.readlines()
#     for n,line in enumerate(aaa):
#         lis = line.replace('\n','').split(',')
#         ls_np = np.array(lis)
#         ls_np = ls_np[:,np.newaxis]
#         ax = sns.heatmap(ls_np)
#         fig = ax.get_figure()
#         fig.savefig(f'heatmap/{n}.png')


a = torch.load('logs/attn_w.pth')
a = a.detach().cpu().numpy().tolist()
# print(a[0].shape)
for i in range(1400):
    plt.figure(figsize=(200,200))
    data = a[i]
    sns.heatmap(data,cbar_kws={'orientation': 'horizontal',"shrink": 0.3},xticklabels=False, yticklabels=False)
    # fig = aaa.get_figure()
    # fig.savefig(f'heatmap/{i}.png')
    # fig.cla()
    plt.savefig(f'heatmap/{i}.png')
    plt.cla()